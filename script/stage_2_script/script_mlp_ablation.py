from local_code.stage_2_code.Dataset_Loader import Dataset_Loader
from local_code.stage_2_code.Method_MLP_Ablation import Method_MLP_Ablation
from local_code.stage_2_code.Result_Saver import Result_Saver
# from local_code.stage_2_code.Setting_KFold_CV import Setting_KFold_CV
# from local_code.stage_2_code.Setting_Train_Test_Split import Setting_Train_Test_Split
from local_code.stage_2_code.Evaluate_Accuracy import Evaluate_Accuracy
from local_code.stage_2_code.plot_utils import plot_learning_curve
import numpy as np
import torch

#---- Multi-Layer Perceptron script ----
if 1:
    #---- parameter section -------------------------------
    np.random.seed(2)
    torch.manual_seed(2)
    #------------------------------------------------------

    # ---- objection initialization setction ---------------
    data_obj = Dataset_Loader('stage_2_dataset', '')
    data_obj.dataset_source_folder_path = '../../data/stage_2_data/'

    method_obj = Method_MLP_Ablation('ablation - high learning rate', '')

    result_obj = Result_Saver('saver', '')
    result_obj.result_destination_folder_path = '../../result/stage_2_result/Ablation_HighLR_'
    result_obj.result_destination_file_name = 'prediction_result'

    #setting_obj = Setting_Train_Test_Split('train test split', '')
    # in_Test_Split('train test split', '')

    evaluate_obj = Evaluate_Accuracy('accuracy', '')
    # ------------------------------------------------------

    # ---- running section ---------------------------------
    print('************ Start ************')
    #setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj)
    #setting_obj.print_setup_summary()
    #mean_score, std_score = setting_obj.load_run_save_evaluate()

    print('--loading data...')
    loaded_data = data_obj.load()

    print('--training and testing...')
    method_obj.data = loaded_data
    learned_result = method_obj.run()
    print('--plotting learning curve...')
    plot_learning_curve(method_obj.loss_history, '../../result/stage_2_result/Ablation_HighLR_learning_curve.png',
    'Ablation - High Learning Rate')

    print('--saving results...')
    result_obj.data = learned_result
    result_obj.save()

    print('--evaluating...')
    evaluate_obj.data = learned_result
    score = evaluate_obj.evaluate()

    print('************ Overall Performance ************')
    print('MLP Accuracy: ' + str(score))
    print('************ Finish ************')
    # ------------------------------------------------------
    

    