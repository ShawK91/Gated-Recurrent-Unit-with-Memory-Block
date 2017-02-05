import sys, numpy as np, math, os
sampling = 50
classifier_runs = 1000
recall_runs = 10000


all_folders = [name for name in os.listdir(".") if os.path.isdir(name)]
data = []
final_save_name = 'ph'
#Determine what type
try:
    filename =  all_folders[0] + '/RSeq_Recall/hof_mem_seq_recall.csv'
    temp = np.loadtxt(filename, delimiter=',')
    data = [[] for i in range(recall_runs / 10)]
    file_tag = '/RSeq_Recall/hof_mem_seq_recall.csv'
    final_save_name = 'recall.csv'
    decorator = np.arange(0,recall_runs, 10 )
except:
    filename = all_folders[0] + '/RSeq_classifier/hof_mem_seq_classifier.csv'
    temp = np.loadtxt(filename, delimiter=',')
    data = [[] for i in range(classifier_runs / 10)]
    file_tag = '/RSeq_classifier/hof_mem_seq_classifier.csv'
    final_save_name = 'classifier.csv'
    decorator = np.arange(0, classifier_runs, 10)

#decorator = temp[:,0]
for i, folder in enumerate(all_folders):
    filename = folder + file_tag
    temp = np.loadtxt(filename, delimiter=',')

    for j in range(len(data)):
        try: data[j].append(temp[j,1])
        except: None

average = [] * len(data)
sd = [] * len(data)
for epoch in data:
    ig = np.array(epoch)
    average.append(np.mean(ig))
    sd.append(np.std(ig)/math.sqrt(len(epoch)))

average = np.reshape(np.array(average), (len(average), 1))
sd = np.reshape(np.array(sd), (len(sd), 1))
decorator = np.reshape(decorator, (len(decorator), 1))

final = np.concatenate((decorator, average), axis=1)
final = np.concatenate((final, sd), axis=1)
save_last = np.reshape(final[-1], (1,3))
final = final[0::sampling]
final = np.concatenate((final, save_last))
#Subsample


np.savetxt(final_save_name, final, delimiter=',' )

































