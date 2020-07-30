import json
import urllib.request, urllib.parse
import os
import numpy as np
import kmeans

f = open('test_data.json', encoding='utf-8')
data = json.load(f)
data_items = data.items()

# id数组
case_id_arr = []

for k, v in data_items:
    for arr in v['cases']:
        if case_id_arr.count(arr['case_id']) == 0:
            case_id_arr.append(arr['case_id'])
# 上传次数
case_upload_times_arr = np.zeros(len(case_id_arr))
# 总分
case_total_score = np.zeros(len(case_id_arr))
# 耗时
case_using_time = np.zeros(len(case_id_arr))
for k, v in data_items:
    for arr in v['cases']:
        index = case_id_arr.index(arr['case_id'])
        case_upload_times_arr[index] = case_upload_times_arr[index] + len(arr['upload_records'])
        case_total_score[index] = case_total_score[index] + arr['final_score']
        if len(arr['upload_records']) > 0:
            case_using_time[index] = case_using_time[index] + arr['upload_records'][len(arr['upload_records']) - 1][
                'upload_time'] - arr['upload_records'][0]['upload_time']


#print(case_upload_times_arr)
case_upload_times_arr=1/case_upload_times_arr
#print(case_upload_times_arr)

#归一化
normalized_case_upload_times_arr=case_upload_times_arr/(np.max(case_upload_times_arr))
#normalized_case_upload_times_arr=case_upload_times_arr
normalized_case_total_score=case_total_score/(np.max(case_total_score))
#normalized_case_total_score=case_total_score
normalized_case_using_time=case_using_time/(np.max(case_using_time))


print(np.concatenate((normalized_case_upload_times_arr,normalized_case_total_score),axis=0))
print(np.hstack((normalized_case_upload_times_arr,normalized_case_total_score)))
dataset=np.reshape(np.concatenate((normalized_case_upload_times_arr,normalized_case_total_score),axis=0),(len(case_id_arr),2))

n=int(input("请输入分类数：(不超过7)"))
centroid_arr = kmeans.random_choose_centroid(dataset, n)
classify_result = kmeans.classify_cluster(dataset, centroid_arr)
#kmeans.show_cluster(centroid_arr, classify_result)
old_var=1
new_var=kmeans.get_variance(centroid_arr,classify_result)
while abs(new_var-old_var)>=0.00001:
    centroid_arr=kmeans.choose_new_centroid(classify_result)
    classify_result=kmeans.classify_cluster(dataset,centroid_arr)
    old_var=new_var
    new_var=kmeans.get_variance(centroid_arr,classify_result)
    #kmeans.show_cluster(centroid_arr,classify_result)
kmeans.show_cluster(centroid_arr, classify_result)
print('各聚类中心倒点：',centroid_arr)
print('聚类结果：',classify_result)
