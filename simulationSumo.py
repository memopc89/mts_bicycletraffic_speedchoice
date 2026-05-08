import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime
import traci
from scipy.constants import g
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
cd = os.getcwd()


def getKinEnergy(ekin, p, M, crr, cda, G, W, rho):
    """
    Computes changes in kinetic energy based on a simplified model by Martin et al. (1998).
    :param ekin:       kinetic energy [J]
    :param p:          power output [W]
    :param M:          mass of inertia [kg]
    :param crr:        rolling friction coefficient
    :param cda:        air drag coefficient
    :param G:          gradient [decimals]
    :param W:          wind speed [m/s] [signed with the direction of travelling]
    :param rho:        air density [kg-m3]
    :return dekin_dt:  derivative of kinetic energy with respect to time
    """
    m = M - 0.14 / (0.36 ** 2)
    n = 0.975      # Chain efficiency
    bearing_fr = 0.091 + 0.0087 * np.sqrt(2 * ekin / M)     # Units: - N*m - N*m*s
    F = 0.5 * rho * cda * ((np.sqrt(2 * ekin / M) + W) ** 2) + m * g * (G + crr) + bearing_fr
    dekin_dt = n * p - F * np.sqrt(2 * ekin / M)

    return dekin_dt


def getBeta(id, model_idx, model_data, coef_names, include_random):
    """
    Get individual coefficients from mixed-model
    :param id:              trajectory id
    :param model_idx:       model index
    :param model_data:      models
    :param coef_names       specify coefficients
    :param include_random   dict, is random effect included? [True or False]
    :return:                beta (dictionary)
    """

    beta = {}
    fixed = model_data.loc[model_idx, 'FixedEffects']
    random = model_data.loc[model_idx, 'RandomEffects'][id]

    # Desired power
    beta['Intercept'] = fixed['Intercept'] + random.get('Group', 0)

    # Coefficients
    for coef in coef_names:
        if coef in fixed.index:
            val = fixed[coef]
            if include_random.get(coef, True) and coef in random.index:
                val += random.get(coef, 0)
            beta[coef] = val

    return beta


def create_demand(bike, i_chars, route, desSpeed, departPos, cfm, path_network):
    with open(path_network + '\\osm_single.rou.xml', 'w') as f:
        f.write(f'<?xml version="1.0" encoding="UTF-8"?>\n')
        f.write(f'<!-- generated on {str(datetime.now())}-->\n')
        f.write(f'\n')
        f.write(f'<routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/routes_file.xsd">\n')
        f.write(f'    <!-- Define bicycle vehicle type -->\n')
        f.write(f'    <vType id="bike" vClass="bicycle" desiredMaxSpeed="{desSpeed}" maxSpeed="20"'
                f' lcStrategic="0" lcCooperative="0" length="2" accel="1.2" decel="3.0" emergencyDecel="3.0" '
                f'sigma="0.0" latAlignment="center" carFollowModel="{cfm}"/>\n')
        f.write(f'\n')
        f.write(f'    <!-- Define route -->\n')
        f.write(f'    <route id="r_0" edges="{route}"/>\n')
        f.write(f'\n')
        f.write(f'    <!-- Define demand -->\n')
        f.write(f'    <vehicle id="{bike}" type="bike" route="r_0" depart="0.00" '
                f'speedFactor="{i_chars["speed_factor"]:.4f}" departSpeed="0" departPos="{departPos}"/>\n')
        f.write(f'</routes>\n')


def run_simulation(sumo_binary, sumo_config, path_network, sumo_output,
                   location, rt_edges, which_grade, grade_dist, inter, utransition, wind_dict,
                   seq_dict, chars, route, desSpeed, departPos, cfm, resparam_avg, weight_avg,
                   kraussPS_approach, speedDG_approach, speedME_approach, powerME_approach,
                   coef_names, include_random, MEmodel):
    print(f'Simulations starts at {str(datetime.now())}')

    df = []
    df_sim_duration = []
    previous_seq = {}
    override_done = set()
    for bike in chars.keys():
        start = datetime.now()

        # Get features of the cyclist
        i_chars = chars[bike]
        v_max = i_chars['speed_max']
        v_des = i_chars['speed_mean']
        p_max = i_chars['power_max']
        if not resparam_avg:
            cda = i_chars['cda_mean']
            crr = i_chars['crr_mean']
        else:
            cda, crr = resparam_avg
        M = i_chars['M'] if not weight_avg else weight_avg
        rho = i_chars['air_density']
        elev_gain = {}

        # Create demand file
        create_demand(bike, i_chars, route, desSpeed, departPos, cfm, path_network)

        # Start simulation
        traci.start([sumo_binary, '-c', sumo_config, '--step-length', '0.1'])
        dt = traci.simulation.getDeltaT()
        try:
            while traci.simulation.getMinExpectedNumber() > 0:
                current_time = traci.simulation.getTime()
                traci.simulationStep()

                for i in traci.vehicle.getIDList():
                    try:

                        # Get current state
                        edge_id = traci.vehicle.getRoadID(i)
                        x, y, z = traci.vehicle.getPosition3D(i)
                        d = traci.vehicle.getDistance(i)
                        v = traci.vehicle.getSpeed(i)
                        a = traci.vehicle.getAcceleration(i)
                        a_max = traci.vehicletype.getAccel('bike')
                        d_max = traci.vehicletype.getDecel('bike')

                        # Get route features
                        _, idx = rt_edges[edge_id]['tree'].query([x, y, z])
                        km, grade, curvature = tuple(rt_edges[edge_id]['params'][idx])
                        G = np.tan(np.deg2rad(traci.vehicle.getSlope(i))) if which_grade == 'GradeSUMO' else grade
                        E = elev_gain.setdefault(int(i), {'in_uphill': False, 'elev_gain': 0, 'prevZ': z})
                        if G > 0.01 and not E['in_uphill']:
                            E['in_uphill'] = True
                        if G > 0.01 and E['in_uphill']:
                            E['elev_gain'] += z - E['prevZ']
                        if G <= 0.01 and E['in_uphill']:
                            E['in_uphill'] = False
                            E['elev_gain'] = 0
                        E['prevZ'] = z
                        ut = 1 if ((utransition['km_min'] <= km) & (km <= utransition['km_max'])).any() else 0
                        wind = np.interp(km, np.array(wind_dict[int(i)])[:, 0], np.array(wind_dict[int(i)])[:, 1])
                        inter_all = 1 if ((inter['km_min'] <= km) & (km <= inter['km_max'])).any() else 0
                        inter_bike = 1 if ((inter['km_min'] <= km) & (km <= inter['km_max'])
                                           & inter['intersection_bike']).any() else 0
                        inter_car = 1 if ((inter['km_min'] <= km) & (km <= inter['km_max'])
                                          & inter['intersection_car']).any() else 0
                        inter_car_signals = 1 if ((inter['km_min'] <= km) & (km <= inter['km_max'])
                                                  & inter['intersection_car_signals']).any() else 0
                        infra = {
                            'grade_uphill': np.abs(np.max([G, 0]))*100,
                            'grade_downhill': np.abs(np.min([G, 0]))*100,
                            'altitude_diff_uphill': E['elev_gain'],
                            'is_uphill_ahead[T.True]': 1 if (ut == 1) and (G < 0.01) else 0,
                            'curvature_rt': curvature,
                            'intersection_all[T.True]': inter_all,
                            'intersection_bike[T.True]': inter_bike,
                            'intersection_car[T.True]': inter_car,
                            'intersection_car_signals[T.True]': inter_car_signals,
                            'wind_speed': wind,
                            'headwind': np.abs(np.max([wind, 0])),
                            'tailwind': np.abs(np.min([wind, 0])),
                        }

                        # Re-starts speed after constrained riding (constrained riding is out of scope)
                        seq_thresholds = seq_dict.get(int(i), [])
                        s = 0
                        v_ini = None
                        for seq_id, km_ini, speed_ini in seq_thresholds:
                            if km >= km_ini:
                                s = seq_id
                                v_ini = speed_ini
                        s_prev = previous_seq.get(int(i), -1)
                        if s != s_prev:
                            if (i, s) not in override_done and v_ini is not None:
                                override_done.add((i, s))
                            previous_seq[int(i)] = s

                        # Speed choice
                        p_est = 0
                        ekin_est = 0
                        v_next = v_des

                        # PowerME-based approach
                        if powerME_approach:
                            traci.vehicle.setSpeedMode(i, 0)  # Disable SUMO speed control

                            # Get current kinetic energy
                            ekin = 0.5 * M * (v ** 2)
                            ekin_min = 0

                            # Get coefficients
                            beta = getBeta(int(i), location, MEmodel, coef_names, include_random)
                            if 'grade_uphill' in beta.keys() and include_random['grade_uphill']:
                                beta['grade_uphill'] = np.max([0, beta['grade_uphill']])
                            if 'grade_downhill' in beta.keys() and include_random['grade_downhill']:
                                beta['grade_downhill'] = np.min([0, beta['grade_downhill']])
                            if (location == 1) and 'altitude_diff_uphill' in beta.keys() and include_random['altitude_diff_uphill']:
                                beta['altitude_diff_uphill'] = np.min([0, beta['altitude_diff_uphill']])
                            if 'wind_speed' in beta.keys() and include_random['wind_speed']:
                                beta['wind_speed'] = np.max([0, beta['wind_speed']])
                            if 'headwind' in beta.keys() and include_random['headwind']:
                                beta['headwind'] = np.max([0, beta['headwind']])
                            if 'tailwind' in beta.keys() and include_random['tailwind']:
                                beta['tailwind'] = np.min([0, beta['tailwind']])

                            # Estimate power
                            p_est = beta.get('Intercept', 0)
                            for coef, value in beta.items():
                                if coef in infra:
                                    p_est += value * infra[coef]
                            p_est = np.max([0, np.min([p_est, p_max])])

                            # Bike dynamics
                            dekin_dt = getKinEnergy(ekin, p_est, M, crr, cda, G, wind, rho)
                            ekin_est = np.max([ekin_min, ekin + dekin_dt * dt])
                            v_next = np.max([0, np.min([np.sqrt((2 * ekin_est) / M), v_max])])

                            # Update speed for next time step
                            traci.vehicle.setSpeed(i, v_next)

                        # SpeedME-based approach
                        if speedME_approach:
                            traci.vehicle.setSpeedMode(i, 0)  # Disable SUMO speed control

                            # Get coefficients
                            beta = getBeta(int(i), location, MEmodel, coef_names, include_random)
                            if 'grade_uphill' in beta.keys() and include_random['grade_uphill']:
                                beta['grade_uphill'] = np.min([0, beta['grade_uphill']])
                            if 'grade_downhill' in beta.keys() and include_random['grade_downhill']:
                                beta['grade_downhill'] = np.max([0, beta['grade_downhill']])
                            if (location == 1) and 'altitude_diff_uphill' in beta.keys() and include_random['altitude_diff_uphill']:
                                beta['altitude_diff_uphill'] = np.min([0, beta['altitude_diff_uphill']])
                            if 'is_uphill_ahead[T.True]' in beta.keys() and include_random['is_uphill_ahead[T.True]']:
                                beta['is_uphill_ahead[T.True]'] = np.max([0, beta['is_uphill_ahead[T.True]']])
                            if 'wind_speed' in beta.keys() and include_random['wind_speed']:
                                beta['wind_speed'] = np.min([0, beta['wind_speed']])
                            if 'headwind' in beta.keys() and include_random['headwind']:
                                beta['headwind'] = np.min([0, beta['headwind']])
                            if 'headwind' in beta.keys() and include_random['tailwind']:
                                beta['tailwind'] = np.max([0, beta['tailwind']])

                            # Estimate speed
                            v_est = beta.get('Intercept', 0)
                            for coef, value in beta.items():
                                if coef in infra:
                                    v_est += value * infra[coef]
                            if v_est >= v:
                                v_amax = v + a_max * dt
                                v_est_max = np.min([v_est, v_amax])
                            elif v_est < v:
                                v_amax = v - d_max * dt
                                v_est_max = np.max([v_est, v_amax])
                            v_next = np.max([0, np.min([v_est_max, v_max])])

                            # Update speed for next time step
                            traci.vehicle.setSpeed(i, v_next)

                        # SpeedDG-based approach
                        if speedDG_approach:
                            traci.vehicle.setSpeedMode(i, 0)  # Disable SUMO speed control

                            # Find new distribution and compute percentile
                            G_bins = np.sort(grade_dist.index.get_level_values('grade_cat').unique())
                            G_idx = np.min([np.searchsorted(G_bins, G * 100), len(G_bins) - 1])
                            G_upper, G_lower = G_bins[G_idx], G_bins[G_idx - 1]
                            G_lower_ptl = grade_dist[grade_dist.index.get_level_values('grade_cat') == G_lower].quantile(i_chars['speed_ptl'])
                            G_upper_ptl = grade_dist[grade_dist.index.get_level_values('grade_cat') == G_upper].quantile(i_chars['speed_ptl'])
                            v_bin = np.interp(G * 100, [G_lower, G_upper], [G_lower_ptl, G_upper_ptl])
                            if v_bin > v:
                                v_amax = v + a_max * dt
                                v_bin_max = np.min([v_bin, v_amax])
                            elif v_bin < v:
                                v_amax = v - d_max * dt
                                v_bin_max = np.max([v_bin, v_amax])
                            elif v_bin == v:
                                v_bin_max = v_bin
                            v_next = np.max([0, np.min([v_bin_max, v_max])])

                            # Update speed for next time step
                            traci.vehicle.setSpeed(i, v_next)

                        # KraussPS
                        if kraussPS_approach:
                            traci.vehicle.setSpeedMode(i, 0)   # Disable SUMO speed control

                            # Calculate acceleration adjusted by gradient
                            G_rad = np.arctan(G)
                            a_eff = np.max([0, a_max - g * np.sin(G_rad)])

                            # Estimate speed
                            v_est = v + a_eff * dt
                            v_est_max = np.max([np.sqrt(a_eff / a_max) * v_des, v - d_max * dt])
                            v_next = np.max([a_max / 2, np.min([v_est, v_est_max])])

                            # Update speed for next time step
                            traci.vehicle.setSpeed(i, v_next)

                        # Store results
                        record = {
                            'time': current_time,
                            'seq': int(s),
                            'ID': int(i),
                            'km': km,
                            'x': x,
                            'y': y,
                            'z': z,
                            'elev_gain': E['elev_gain'],
                            'distance': d,
                            'speed': v,
                            'speed_target': v_next,
                            'acc': a,
                            'power': p_est,
                            'ekin': ekin_est,
                            'grade': grade,
                            'gradeTraci': np.tan(np.deg2rad(traci.vehicle.getSlope(i))),
                            'curvature': curvature,
                            'intersection_all': inter_all,
                            'intersection_bike': inter_bike,
                            'intersection_car': inter_car,
                            'intersection_car_signals': inter_car_signals,
                            'wind': wind
                        }
                        df.append(record)

                        # Remove cyclist if completely stopped (safety-check; no instances)
                        if (d > 5) and (v <= 0.5):
                            traci.vehicle.remove(i, reason=traci.constants.REMOVE_VAPORIZED)
                            print(f"ID {i} stalls at low speed or fully stops at t={current_time}")
                            continue

                    except traci.TraCIException:
                        continue
        finally:
            print(f'Simulation ends for ID {bike}')
            traci.close()
            stop = datetime.now() - start
            record_sim_duration = {
                'location': location,
                'ID': bike,
                'sim_duration': stop.seconds,
            }
            df_sim_duration.append(record_sim_duration)
    df = pd.DataFrame(df)
    df['location'] = location
    df.set_index(['location', 'ID', 'seq'], inplace=True)
    df_sim_duration = pd.DataFrame(df_sim_duration)
    df_sim_duration.set_index(['location', 'ID'], inplace=True)
    df.to_pickle(sumo_output + '\\SimData_' + ('Linköping.pkl' if location == 0 else 'Wuppertal.pkl'))
    df_sim_duration.to_pickle(sumo_output + '\\SimDuration_' + ('Linköping.pkl' if location == 0 else 'Wuppertal.pkl'))
    print(f'Simulations ends at {str(datetime.now())}')
