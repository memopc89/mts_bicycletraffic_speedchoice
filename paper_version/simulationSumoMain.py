import os
import pandas as pd
import sumolib
from scipy.spatial import KDTree
from simulationSumo import run_simulation
cd = os.getcwd()

# General --------------------------------------------------------------------------------------------------------------
# Configuration for each of the two scenarios: Linköping and Wuppertal
sumo_binary = 'sumo'       # sumo or sumo-gui
which_grade = 'GradeRT'    # which grade data to use: "GradeSUMO" or "GradeRT" or None
route = ["-48773164#10 -48773164#3 -48773164#1 48773167 -36794859#3 -36794859#1 -36794859#0 34001202#0 34001202#1 "
         "34001202#3 27021525#1 27021505#0 27021505#2 27021505#3 27021505#4 8125605#0 36794859#2 -48773167 38990161 "
         "48773164#3 48773164#7 -E1",
         "-423 -412#0 9#4 9#0 -278 -441 -442 -299 -300 -504#0 -515#0 -636#0 -636#3 -108#0 -108#2 -108#5 -530 -649 -651 "
         "-654 -659#0 -659#2 -659#4 -680#2 -9#0 -9#4 412#0"]
departPos = [9, 0]
desSpeed = [6.09, 6.55]
resParam = [[0.7, 0.008], [0.559, 0.008]]
weightM = [103.5, 99.3]
locations = ['Linköping', 'Wuppertal']
intersections_ids = [[1, 2, 3, 5, 6, 7, 8, 9, 12, 13, 14], [1, 2, 3, 4, 5, 6, 7]]
grade_smoothing = [50, 100]

# %% Simulation --------------------------------------------------------------------------------------------------------
krauss_approach = False
kraussPS_approach = False
speedDG_approach = False
speedME_approach = False
powerME_approach = False
for l, l_label in enumerate(locations):

    # Participant characteristics
    chars = pd.read_pickle(cd + '\\Data\\ParamsUser.pkl')
    cond_loc = chars.index.get_level_values('location') == l
    chars['M'] = chars['tot_weight'] + 0.14 / (0.36 ** 2)  # m + I/r^2
    chars = chars[cond_loc].reset_index(level='location').to_dict(orient='index')

    # Trajectory sequences
    trajs = pd.read_pickle(cd + '\\Data\\Data.pkl')
    cond_loc = trajs['location'] == l
    seq_idx = trajs[cond_loc].groupby(['ID', 'seq'])['km'].idxmin()
    seq = trajs[cond_loc].loc[seq_idx, ['ID', 'seq', 'km', 'speed']]
    seq_dict = {}
    for (i, seq_id) in seq.groupby('ID'):
        seq_dict[i] = seq_id.sort_values('km')[['seq', 'km', 'speed']].values.tolist()

    # Wind data (space-based)
    trajs_full = pd.read_pickle(cd + '\\Data\\DataFull.pkl')
    trajs_full.loc[trajs_full['location'] == 1, 'wind_speed'] = 0
    wind_dict = {}
    for (i, wind_id) in trajs_full.groupby('ID'):
        wind_dict[i] = wind_id.sort_values('km')[['km', 'wind_speed']].values.tolist()

    # Speed distributions by gradient bins
    grade_dist = pd.read_pickle(cd + '\\Data Analysis\\Grade\\CharsGrade.pkl')
    grade_dist = grade_dist.loc[grade_dist.index.get_level_values('location') == l, 'speed_mean']
    grade_dist.name = 'speed_mean'
    flat_dist = pd.read_pickle(cd + '\\Data\\ParamsUser.pkl')
    flat_dist = flat_dist[flat_dist.index.get_level_values('location') == l]
    flat_dist['grade_cat'] = 0
    flat_dist = flat_dist.set_index('grade_cat', append=True)
    grade_dist.update(flat_dist['speed_mean'])

    # Route SUMO data
    path_network = cd + '\\Simulation\\SUMO_Networks\\' + l_label
    rt_sumo = pd.read_pickle(path_network + '\\route.pkl')
    rt_sumo['gradeS'] = rt_sumo['gradeS'].rolling(window=grade_smoothing[l], center=True, min_periods=1).mean()
    rt_edges = {}
    for edge_id, group in rt_sumo.groupby('ID'):
        points = group[['x', 'y', 'z']].values
        rt_edges[edge_id] = {
            'tree': KDTree(points),
            'params': group[['km', 'gradeS', 'kS']].values
        }

    # Intersection data (space-based)
    inter = pd.read_pickle(
        cd + '\\Data Analysis\\Intersection\\IntersectionSegments_' + ('Lkpg' if l == 0 else 'Wupp') + '.pkl')
    inter['km_min'] = inter['km_min'].round(3)
    inter['km_max'] = inter['km_max'].round(3)
    inter = inter.loc[inter.index.isin(intersections_ids[l])]

    # U_transitions data (space-based)
    utransition = pd.read_pickle(cd + '\\Data Analysis\\Grade\\GradeSegmentsUTransition_' + ('Lkpg' if l == 0 else 'Wupp') + '.pkl')
    utransition.drop(index=[1, 6], axis=0, inplace=True) if l == 0 else utransition
    utransition['km_min'] = utransition['km_min'].round(3)
    utransition['km_max'] = utransition['km_max'].round(3)

    # Default or SpeedDG approach
    if krauss_approach or kraussPS_approach or speedDG_approach:
        cfm = 'Krauss'
        if krauss_approach:
            folder_output = 'Model_Krauss'
        elif kraussPS_approach:
            folder_output = 'Model_KraussPS'
        elif speedDG_approach:
            folder_output = 'Model_SpeedDG'

        # Run
        sumo_output = cd + '\\Simulation\\' + folder_output
        os.makedirs(sumo_output, exist_ok=True)
        sumo_config = path_network + '\\osm.sumocfg'
        sumo_net = sumolib.net.readNet(path_network + '\\osm_patched.net.xml', withInternal=True)
        print(f'---------MODEL STARTS----------')
        run_simulation(sumo_binary, sumo_config, path_network, sumo_output,
                       l, rt_edges, which_grade, grade_dist, inter, utransition, wind_dict,
                       seq_dict, chars, route[l], desSpeed[l], departPos[l], cfm, [], [],
                       kraussPS_approach, speedDG_approach, speedME_approach, powerME_approach,
                       [], [], [])
        print(f'---------MODEL ENDS----------')

    # Speed MixedEffects approach
    if speedME_approach:
        cfm = 'Krauss'
        fe = ['grade_uphill', 'grade_downhill', 'altitude_diff_uphill', 'is_uphill_ahead[T.True]', 'curvature_rt',
              'intersection_all[T.True]', 'wind_speed']
        re = ['grade_uphill', 'grade_downhill', 'altitude_diff_uphill', 'is_uphill_ahead[T.True]', 'wind_speed']
        models_no = [17, 19]
        models_paths = [cd + '\\Simulation\\Model_SpeedME\\MEM\\Speed_model_O_Gfr_CI_Wfr.pkl',
                        cd + '\\Simulation\\Model_SpeedME\\MEM\\Speed_model_O_Gfr_CI.pkl']
        for m in models_no:
            folder_output = f'Model_SpeedME\\Model{int(m)}'

            # Get model
            include_random = {r: False for r in fe} if m == 16 else {r: True for r in fe}
            MEmodel = pd.read_pickle(models_paths[l])

            # Run
            sumo_output = cd + '\\Simulation\\' + folder_output
            os.makedirs(sumo_output, exist_ok=True)
            sumo_config = path_network + '\\osm.sumocfg'
            sumo_net = sumolib.net.readNet(path_network + '\\osm_patched.net.xml',  withInternal=True)
            print(f'---------MODEL {int(m)} STARTS----------')
            run_simulation(sumo_binary, sumo_config, path_network, sumo_output,
                           l, rt_edges, which_grade, grade_dist, inter, utransition, wind_dict,
                           seq_dict, chars, route[l], desSpeed[l], departPos[l], cfm, [], [],
                           kraussPS_approach, speedDG_approach, speedME_approach, powerME_approach,
                           fe, include_random, MEmodel)
            print(f'---------MODEL {int(m)} ENDS----------')

    # Power MixedEffects approach
    if powerME_approach:
        cfm = 'Krauss'
        fe = ['grade_uphill', 'grade_downhill', 'altitude_diff_uphill', 'is_uphill_ahead[T.True]', 'curvature_rt',
              'intersection_all[T.True]', 'wind_speed']
        re = ['grade_uphill', 'grade_downhill', 'altitude_diff_uphill', 'is_uphill_ahead[T.True]', 'wind_speed']
        models_no = [16, 17, 18, 19]
        models_paths = [cd + '\\Simulation\\Model_PowerME\\MEM\\Power_model_OE_Gfr_CI_Wfr.pkl',
                        cd + '\\Simulation\\Model_PowerME\\MEM\\Power_model_OE_Gfr_CI.pkl']
        for m in models_no:
            folder_output = f'Model_PowerME\\Model{int(m)}'

            # Get model
            include_random = {r: False for r in fe} if m <= 17 else {r: True for r in fe}
            resparam_avg = [] if m == 17 or m == 19 else resParam[l]
            weight_avg = [] if m == 17 or m == 19 else weightM[l]
            MEmodel = pd.read_pickle(models_paths[l])

            # Run
            sumo_output = cd + '\\Simulation\\' + folder_output
            os.makedirs(sumo_output, exist_ok=True)
            sumo_config = path_network + '\\osm.sumocfg'
            sumo_net = sumolib.net.readNet(path_network + '\\osm_patched.net.xml', withInternal=True)
            print(f'---------MODEL {int(m)} STARTS----------')
            run_simulation(sumo_binary, sumo_config, path_network, sumo_output,
                           l, rt_edges, which_grade, grade_dist, inter, utransition, wind_dict,
                           seq_dict, chars, route[l], desSpeed[l], departPos[l], cfm, resparam_avg, weight_avg,
                           kraussPS_approach, speedDG_approach, speedME_approach, powerME_approach,
                           fe, include_random, MEmodel)
            print(f'---------MODEL {int(m)} ENDS----------')
