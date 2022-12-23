import itertools
import json
from matplotlib import font_manager
import matplotlib.pyplot as plt
import numpy as np

def compare_success(env_id:str):
    if env_id.lower() == 'reach':
        # ---------------------------------------------------------   LOAD FILES   -------------------------------------------------------------------------------------------------

        # file='tmp/checkpoints/ddpg_train_info.json'
        file='tmp/train_res/202212062033_PandaReach-v3_ddpg_train_vals.json'
        with open(file) as rf:
            dict_res_ddpg_1 = json.load(rf)

        # file='tmp/checkpoints/ddpg_train_info.json'
        file='tmp/train_res/202212062121_PandaReach-v3_ddpg_train_vals.json'
        with open(file) as rf:
            dict_res_ddpg_no_bias = json.load(rf)
            
        # file='tmp/checkpoints/ddpg_train_info.json'
        file='tmp/train_res/202212062207_PandaReach-v3_ddpg_train_vals.json'
        with open(file) as rf:
            dict_res_ddpg_3 = json.load(rf)
            
        # file='tmp/checkpoints/ddpg_train_info.json'
        file='tmp/train_res/202212062254_PandaReach-v3_ddpg_train_vals.json'
        with open(file) as rf:
            dict_res_ddpg_4 = json.load(rf)
            
        # file='tmp/checkpoints/ddpg_train_info.json'
        file='tmp/train_res/202212062341_PandaReach-v3_ddpg_train_vals.json'
        with open(file) as rf:
            dict_res_ddpg_5 = json.load(rf)
            
        # ---------------------------------------------------------   LOAD FILES   -------------------------------------------------------------------------------------------------

        # ---------------------------------------------------------   PREPARE DATA -------------------------------------------------------------------------------------------------

        lists = []
        lists.append(list(np.array(dict_res_ddpg_1['success_rate'], dtype=float)))
        lists.append(list(np.array(dict_res_ddpg_no_bias['success_rate'], dtype=float)))
        lists.append(list(np.array(dict_res_ddpg_3['success_rate'], dtype=float)))
        lists.append(list(np.array(dict_res_ddpg_4['success_rate'], dtype=float)))
        lists.append(list(np.array(dict_res_ddpg_5['success_rate'], dtype=float)))

        X1 = np.array([x for x in itertools.zip_longest(*lists, fillvalue=0)], dtype=float)
        mu1 = np.mean(X1, axis=1)
        median = np.median(X1, axis=1)
        sigma1 = np.std(X1, axis=1)

        # ---------------------------------------------------------   PREPARE DATA -------------------------------------------------------------------------------------------------

        # ---------------------------------------------------------   PLOT DATA    -------------------------------------------------------------------------------------------------

        '''PLOT success rates'''    
        font_prop = font_manager.FontProperties(size=8)  

        # plt.plot([float(p) for p in dict_res_ddpg_1['success_rate_ts']], [float(d) for d in dict_res_ddpg_normNoi['success_rate']], 
        #          label='DDPG - sigma=0.2, gamma=0.99, polyak=0.95, HER, random_prop=0.3, nn_dims=[256,256,256]', 
        #          color=(0., 0., 1., 1.))
        plt.plot([float(p) for p in dict_res_ddpg_1['success_rate_ts']], median, 
                label='DDPG - sigma=0.2, gamma=0.99, polyak=0.95, HER, random_prop=0.3, nn_dims=[256,256,256]', 
                color=(0., 0., 1., 1.))
        plt.fill_between([float(p) for p in dict_res_ddpg_1['success_rate_ts']], (mu1+sigma1).clip(0., 1.), (mu1-sigma1).clip(0., 1.), color=(0., 0., 1., 0.3),
                        label='Mean +/- 1 STD') #facecolor='blue', alpha=0.5)

        # plt.plot([float(d) for d in dict_res_ddpg_normNoi_2['success_rate']], label='DDPG - sigma=0.05, gamma=0.99, polyak=0.95, HER, random_prop=0.2, nn_dims=[256,256,256]', 
        #          color=(1., 0., 0., 0.75))
        # plt.plot([float(d) for d in dict_res_ddpg_rand_prop_3['success_rate']], label='DDPG - sigma=0.3, gamma=0.97, polyak=0.995, HER, random_prop=0.3, nn_dims=[64,64,64]', 
        #          color=(0., 1., 0., 0.5))

        plt.title('Reach - Success Rate per Time step')
        plt.xlabel(f'Time steps')
        plt.ylabel('Success Rate')
        plt.ylim(bottom=-0.05, top=1.15)
        plt.legend(loc="upper left", prop=font_prop)
        plt.show()
    elif env_id.lower() == 'pickandplace':
        # ---------------------------------------------------------   LOAD FILES   -------------------------------------------------------------------------------------------------

        # file='tmp/checkpoints/ddpg_train_info.json'
        file='tmp/train_res/202212161500_PandaPickAndPlace-v3_ddpg_train_vals.json'
        with open(file) as rf:
            dict_res_ddpg_1 = json.load(rf)

        # file='tmp/ddpg_train_info.json'
        # file='tmp/checkpoints/ddpg_train_info.json'
        file='tmp/train_res/202212151236_PandaPickAndPlace-v3_ddpg_train_vals.json'
        with open(file) as rf:
            dict_res_ddpg_no_bias = json.load(rf)
            
        # file='tmp/checkpoints/ddpg_train_info.json'
        file='tmp/train_res/202212170803_PandaPickAndPlace-v3_ddpg_train_vals.json'
        with open(file) as rf:
            dict_res_ddpg_3 = json.load(rf)
            
        # file='tmp/checkpoints/ddpg_train_info.json'
        file='tmp/train_res/202212190844_PandaPickAndPlace-v3_ddpg_train_vals.json' # initialize -> all [-1,1]
        with open(file) as rf:
            dict_res_ddpg_4 = json.load(rf)
        
        # file='tmp/checkpoints/ddpg_train_info.json'
        file='tmp/train_res/202212230634_PandaPickAndPlace-v3_ddpg_train_vals.json'
        with open(file) as rf:
            dict_res_ddpg_5 = json.load(rf)
            
        file='tmp/checkpoints/ddpg_train_info.json'
        # file='tmp/train_res/202212220910_PandaPickAndPlace-v3_ddpg_train_vals.json'
        with open(file) as rf:
            dict_res_ddpg_6 = json.load(rf)
            
            
        file='tmp/checkpoints/td3_train_info.json'
        # file='tmp/train_res/202212220910_PandaPickAndPlace-v3_ddpg_train_vals.json'
        with open(file) as rf:
            dict_res_td3_1 = json.load(rf)
            
        # ---------------------------------------------------------   LOAD FILES   -------------------------------------------------------------------------------------------------

        # ---------------------------------------------------------   PREPARE DATA -------------------------------------------------------------------------------------------------

        x_points = [float(p) for p in dict_res_ddpg_1['success_rate_ts'] if p <= int(1.15e6)]
        x_points_2 = [float(p) for p in dict_res_ddpg_5['success_rate_ts'] if p <= int(1.15e6)]
        lists = []
        lists.append(list(np.array(dict_res_ddpg_1['success_rate'][:min(len(x_points),len(x_points_2))], dtype=float)))
        # lists.append(list(np.array(dict_res_ddpg_2['success_rate'][:len(x_points)], dtype=float)))
        # lists.append(list(np.array(dict_res_ddpg_3['success_rate'][:min(len(x_points),len(x_points_2))], dtype=float)))
        # lists.append(list(np.array(dict_res_ddpg_4['success_rate'][:len(x_points)], dtype=float)))
        lists.append(list(np.array(dict_res_ddpg_5['success_rate'][:min(len(x_points),len(x_points_2))], dtype=float)))
        lists.append(list(np.array(dict_res_ddpg_6['success_rate'][:min(len(x_points),len(x_points_2))], dtype=float)))

        X1 = np.array([x for x in itertools.zip_longest(*lists, fillvalue=0)], dtype=float)
        mu1 = np.mean(X1, axis=1)
        median = np.median(X1, axis=1)
        sigma1 = np.std(X1, axis=1)

        # ---------------------------------------------------------   PREPARE DATA -------------------------------------------------------------------------------------------------

        # ---------------------------------------------------------   PLOT DATA    -------------------------------------------------------------------------------------------------

        '''PLOT success rates'''    
        font_prop = font_manager.FontProperties(size=8)  

        value1 = [float(d) for d in dict_res_ddpg_1['success_rate'][:len(x_points)]]
        plt.plot(x_points, value1, 
                label='DDPG - sigma=0.2, gamma=0.98, polyak=0.95, HER, random_prop=0.3, nn_dims=[256,256,256], init_bias=uniform(-2, 2), Batch Normalization = LayerNorm. Iter 1', 
                color=(0.4, 0.4, 0.4, 1.))
        # plt.plot(x_points if len(x_points) < len(x_points_2) else x_points_2, median, 
        #         label='DDPG - median +-1std', 
        #         color=(0., 0., 1., 1.))
        # plt.fill_between(x_points if len(x_points) < len(x_points_2) else x_points_2, (mu1+sigma1).clip(0., 1.), (mu1-sigma1).clip(0., 1.), color=(0., 0., 1., 0.3))#,
        #                 # label='Mean +/- 1 STD') #facecolor='blue', alpha=0.5)
                        
                        
        # value2 = [float(d) for d in dict_res_ddpg_3['success_rate'][:len(x_points)]]
        # plt.plot([float(p) for p in dict_res_ddpg_3['success_rate_ts'][:len(x_points)]], value2, 
        #         label='DDPG - sigma=0.2, gamma=0.98, polyak=0.95, HER, random_prop=0.3, nn_dims=[256,256,256], init_bias=uniform(-4, 4), Batch Normalization = LayerNorm. Iter 2', 
        #         color=(0., 1., 0., 0.6))
        # value3 = [float(d) for d in dict_res_ddpg_4['success_rate'][:len(x_points)]]
        # plt.plot([float(p) for p in dict_res_ddpg_4['success_rate_ts'][:len(x_points)]], value3, 
        #         label='DDPG - sigma=0.2, gamma=0.99, polyak=0.95, HER, random_prop=0.3, nn_dims=[256,256,256], init_bias=uniform(-1, 1), Batch Normalization = LayerNorm. Iter 3', 
        #         color=(1., 0., 1., 0.7))
        value4 = [float(d) for d in dict_res_ddpg_5['success_rate'][:len(x_points)]]
        plt.plot([float(p) for p in dict_res_ddpg_5['success_rate_ts'][:len(x_points)]], value4, 
                label='DDPG - sigma=0.2, gamma=0.99, polyak=0.95, HER, random_prop=0.3, nn_dims=[256,256,256], init_bias=uniform(-.65,.65;-.65,.65;-.65,.65;-.004,.004), '
                'init_weights=uniform(-.07,.07;-.07,.07;-.07,.07), \ninit_out_weights=uniform(-3e3, 3e3), Batch Normalization = LayerNorm. Iter 3', 
                color=(0.89, 0.89, 0., 0.8))
        value5 = [float(d) for d in dict_res_ddpg_6['success_rate'][:len(x_points)]]
        plt.plot([float(p) for p in dict_res_ddpg_6['success_rate_ts'][:len(x_points)]], value5, 
                label='DDPG - sigma=0.2, gamma=0.99, polyak=0.95, HER, random_prop=0.3, nn_dims=[256,256,256], init_bias=uniform(-.65,.65;-.65,.65;-.65,.65;-.004,.004), '
                'init_weights=uniform(-.07,.07;-.07,.07;-.07,.07), \ninit_out_weights=uniform(-3e3, 3e3), Batch Normalization = LayerNorm. Iter 3', 
                color=(1., 0., 0., 0.8))
        
        # -----------------------------------------------  TD3  -----------------
        # td3_1 = [float(d) for d in dict_res_td3_1['success_rate'][:len(x_points)]]
        # plt.plot([float(p) for p in dict_res_td3_1['success_rate_ts'][:len(x_points)]], td3_1, 
        #         label='TD3 - sigma=0.2, gamma=0.99, polyak=0.95, HER, random_prop=0.3, nn_dims=[256,256,256], init_bias=uniform(-.1,.1;-.1,.1;-.1,.1;-.003,.003), '
        #         'init_weights=uniform(-.07,.07;-.07,.07;-.07,.07), \ninit_out_weights=uniform(-3e3, 3e3), Batch Normalization = LayerNorm. Iter 3', 
        #         color=(1., 0., 0., 0.8))




        # value_no_bias = [float(d) for d in dict_res_ddpg_no_bias['success_rate'][:len(x_points)]]
        # plt.plot([float(p) for p in dict_res_ddpg_no_bias['success_rate_ts'][:len(x_points)]], value_no_bias, 
        #          label='DDPG - sigma=0.2, gamma=0.98, polyak=0.95, HER, random_prop=0.3, nn_dims=[256,256,256], no bias', color=(1., 0., 0., 0.5))
        
        
        
        
        
        plt.axhline(y=1.0, color=(0.,0.,0.,1.), linestyle='--')
        # plt.axhline(y=max(median), color=(0.,0.,1.,1.), linestyle=':')
        # plt.axhline(y=max(value_no_bias), color=(1.,0.,0.,1.), linestyle=':')
        # plt.axvline(x=int(1.5e6), color=(0.,0.,0.,1.), linestyle=':')
        plt.title('PickAndPlace - Success Rate per Time step')
        plt.xlabel(f'Time steps')
        plt.ylabel('Success Rate')
        plt.ylim(bottom=-0.05, top=1.15)
        plt.xlim(left=-5000., right=int(1.17e6))
        plt.legend(loc="upper left", prop=font_prop)
        plt.show()
    elif env_id.lower() == 'push':
        # ---------------------------------------------------------   LOAD FILES   -------------------------------------------------------------------------------------------------

        # file='tmp/checkpoints/ddpg_train_info.json'
        file='tmp/train_res/202212130905_PandaPush-v3_ddpg_train_vals.json'
        with open(file) as rf:
            dict_res_ddpg_1 = json.load(rf)

        # file='tmp/checkpoints/ddpg_train_info.json'
        file='tmp/train_res/202212131129_PandaPush-v3_ddpg_train_vals.json'
        with open(file) as rf:
            dict_res_ddpg_no_bias = json.load(rf)
            
        # file='tmp/checkpoints/ddpg_train_info.json'
        file='tmp/train_res/202212131354_PandaPush-v3_ddpg_train_vals.json'
        with open(file) as rf:
            dict_res_ddpg_3 = json.load(rf)
            
        # file='tmp/checkpoints/ddpg_train_info.json'
        # # file='tmp/train_res/202212062254_PandaReach-v3_ddpg_train_vals.json'
        # with open(file) as rf:
        #     dict_res_ddpg_4 = json.load(rf)
            
        # # file='tmp/checkpoints/ddpg_train_info.json'
        # file='tmp/train_res/202212062341_PandaReach-v3_ddpg_train_vals.json'
        # with open(file) as rf:
        #     dict_res_ddpg_5 = json.load(rf)
            
        # ---------------------------------------------------------   LOAD FILES   -------------------------------------------------------------------------------------------------

        # ---------------------------------------------------------   PREPARE DATA -------------------------------------------------------------------------------------------------

        lists = []
        lists.append(list(np.array(dict_res_ddpg_1['success_rate'], dtype=float)))
        lists.append(list(np.array(dict_res_ddpg_no_bias['success_rate'], dtype=float)))
        lists.append(list(np.array(dict_res_ddpg_3['success_rate'], dtype=float)))
        # lists.append(list(np.array(dict_res_ddpg_4['success_rate'], dtype=float)))
        # lists.append(list(np.array(dict_res_ddpg_5['success_rate'], dtype=float)))

        X1 = np.array([x for x in itertools.zip_longest(*lists, fillvalue=0)], dtype=float)
        mu1 = np.mean(X1, axis=1)
        median = np.median(X1, axis=1)
        sigma1 = np.std(X1, axis=1)

        # ---------------------------------------------------------   PREPARE DATA -------------------------------------------------------------------------------------------------

        # ---------------------------------------------------------   PLOT DATA    -------------------------------------------------------------------------------------------------

        '''PLOT success rates'''    
        font_prop = font_manager.FontProperties(size=8)  

        # plt.plot([float(p) for p in dict_res_ddpg_1['success_rate_ts']], [float(d) for d in dict_res_ddpg_normNoi['success_rate']], 
        #          label='DDPG - sigma=0.2, gamma=0.99, polyak=0.95, HER, random_prop=0.3, nn_dims=[256,256,256]', 
        #          color=(0., 0., 1., 1.))
        plt.plot([float(p) for p in dict_res_ddpg_1['success_rate_ts']], median, 
                label='DDPG - sigma=0.2, gamma=0.98, polyak=0.95, HER, random_prop=0.3, nn_dims=[256,256,256]', 
                color=(0., 0., 1., 1.))
        plt.fill_between([float(p) for p in dict_res_ddpg_1['success_rate_ts']], (median+sigma1).clip(0., 1.), (median-sigma1).clip(0., 1.), color=(0., 0., 1., 0.3),
                        label='median +/- 1 STD') #facecolor='blue', alpha=0.5)

        # plt.plot([float(d) for d in dict_res_ddpg_normNoi_2['success_rate']], label='DDPG - sigma=0.05, gamma=0.99, polyak=0.95, HER, random_prop=0.2, nn_dims=[256,256,256]', 
        #          color=(1., 0., 0., 0.75))
        # plt.plot([float(d) for d in dict_res_ddpg_rand_prop_3['success_rate']], label='DDPG - sigma=0.3, gamma=0.97, polyak=0.995, HER, random_prop=0.3, nn_dims=[64,64,64]', 
        #          color=(0., 1., 0., 0.5))

        plt.axhline(y=1.0, color=(0.,0.,0.,1.), linestyle=':')
        plt.title('Push - Success Rate per Time step')
        plt.xlabel(f'Time steps')
        plt.ylabel('Success Rate')
        plt.ylim(bottom=-0.05, top=1.15)
        plt.legend(loc="upper left", prop=font_prop)
        plt.show()



if __name__ == '__main__':
    # env_id = 'Reach'
    env_id = 'PickAndPlace'
    # env_id = 'Push'
    
    compare_success(env_id=env_id)