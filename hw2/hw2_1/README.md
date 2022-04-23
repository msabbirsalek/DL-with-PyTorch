# CPSC8430_HW2: Video Captioning using S2VT

## Steps to follow:

1. Clone the repository
2. For training, make sure that, ModelSaveLoc = "TrainedModels" in the main_execution() function in the Videocaptioning_main.py (set as default). Models in this version is executed for 20 epochs
3. Before running the bash script "hw2_seq2seq.sh", check the "__main__" function and please make sure that, train = False, test=True are given.
5. For testing, make sure that, ModelSaveLoc = "BestModel" in the testmodel() function in the Videocaptioning_main.py (set as deafult), and then run the bash script named "hw2_seq2seq.sh" with "testing_data" and result3.txt. (During Testing)

## Bleu Score for top 5 models are given in the file named "final_result.csv"

## Top 2 models
1. model_batchsize_32_hidsize_128_DP_0.2_worddim_2048.h5 (Bleu Score - 0.699341592)
2. model_batchsize_16_hidsize_512_DP_0.2_worddim_1024.h5 (Bleu Score - 0.690616515)


