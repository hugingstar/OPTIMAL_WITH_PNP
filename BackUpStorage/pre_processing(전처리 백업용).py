import pandas as pd
import datetime
year = 2020
month = 1
day = 15
num_ind = 961
num_otd = 909

indoor_path = "/Users/yslee/PycharmProjects/OPTIMAL/Data/indoor/log_device_sac_indoor_진리관2층_{}.csv".format(num_ind)
outdoor_path = "/Users/yslee/PycharmProjects/OPTIMAL/Data/outdoor_meter/log_device_sac_outdoor_진리관2층_{}.csv".format(num_otd)

df_ind = pd.read_csv(indoor_path)
df_otd = pd.read_csv(outdoor_path)

time = "updated_time"

"""str to datetime"""
print("Indoor shape : {} - Outdoor shape : {}".format(df_ind.shape, df_otd.shape))
df_ind[time] = pd.to_datetime(df_ind[time])
df_ind[time] = df_ind[time].map(lambda x: x.replace(second=0, microsecond=0))
df_otd[time] = pd.to_datetime(df_otd[time])
df_otd[time] = df_otd[time].map(lambda x: x.replace(second=0, microsecond=0))


"""Cutting1 : Datetime"""
start = datetime.datetime(year, month, day, 0, 0, 0)
end = start + datetime.timedelta(days=1)
print("Start - {} End - {}".format(start, end))

df_ind = df_ind[df_ind[time] >= start]
df_ind = df_ind[df_ind[time] < end]
df_ind.sort_values(by=time)

df_otd = df_otd[df_otd[time] >= start]
df_otd = df_otd[df_otd[time] < end]
df_otd.sort_values(by=time)

df_ind = df_ind.set_index(time, drop=True, inplace=False)
df_otd = df_otd.set_index(time, drop=True, inplace=False)
print("Cutting  by time: Indoor shape : {} - Outdoor shape : {}".format(df_ind.shape, df_otd.shape))

"""Index unique check"""
print("Index Unique : Indoor : {} - Outdoor : {} ".format(df_ind.index.is_unique, df_otd.index.is_unique))
if df_ind.index.is_unique or df_otd.index.is_unique is not True: #Timeindex가 하나라도 유니크하지 않으면 중복값 제
    """Duplicated value"""
    df_ind = df_ind[~df_ind.index.duplicated(keep='first')]
    df_otd = df_otd[~df_otd.index.duplicated(keep='first')]
    print("Duplicated remove : Indoor shape : {} - Outdoor shape : {}".format(df_ind.shape, df_otd.shape))
else: #Timeindex 모두가 유니크하면 pass
    pass

"""Save 1 day data"""
# df_ind.to_csv("/Users/yslee/PycharmProjects/OPTIMAL/Results/Indoor_{}.csv".format(num_ind))
# df_otd.to_csv("/Users/yslee/PycharmProjects/OPTIMAL/Results/Outdoor_{}.csv".format(num_otd))
# print(df_ind.columns)
# print(df_otd.columns)

"""Cutting2 : columns Only for analysis"""
df_ind = df_ind[['current_room_temp','indoor_set_temp','indoor_power']]
df_ind.rename(columns = {'current_room_temp' : 'current_room_temp_{}'.format(num_ind)}, inplace = True)
df_ind.rename(columns = {'indoor_set_temp' : 'indoor_set_temp_{}'.format(num_ind)}, inplace = True)
df_ind.rename(columns = {'indoor_power' : 'indoor_power_{}'.format(num_ind)}, inplace = True)
# df_ind.to_csv("/Users/yslee/PycharmProjects/OPTIMAL/Results/Indoor_{}.csv".format(num_ind))

df_otd = df_otd[['outdoor_temperature']]
df_otd.rename(columns = {'outdoor_temperature' : 'outdoor_temperature_{}'.format(num_ind)}, inplace = True)
# df_otd.to_csv("/Users/yslee/PycharmProjects/OPTIMAL/Results/Outdoor_{}.csv".format(num_otd))

print("Columns remove : Indoor shape : {} - Outdoor shape : {}".format(df_ind.shape, df_otd.shape))

"""합치기 : (1440:features) 로 만들"""
df_tem = pd.merge(df_ind, df_otd, left_index=True, right_index=True, how='outer')

col = df_tem.columns
df_col = pd.DataFrame(columns=col)
df_start = pd.DataFrame({time:[start]})
df_start = pd.merge(df_start, df_col, left_index=True, right_index=True, how='outer')
df_start = df_start.set_index(time)
df_end = pd.DataFrame({time:[end]})
df_end = pd.merge(df_end, df_col, left_index=True, right_index=True, how='outer')
df_end = df_end.set_index(time)
df_tem = pd.concat([df_start, df_tem], ignore_index=False)
df_tem = pd.concat([df_tem, df_end], ignore_index=False)
df_tem = df_tem[~df_tem.index.duplicated(keep='first')]

power = df_tem.columns[df_tem.columns.str.contains("indoor_power")].tolist()[0]
df_tem[power] = df_tem[power].str.replace('t', '1')
df_tem[power] = df_tem[power].str.replace('f', '0')
df_tem = df_tem.asfreq('1Min')
df_tem = df_tem.drop(df_tem.index[-1])
df_tem = df_tem.fillna(method='ffill')
df_tem = df_tem.fillna(method='bfill')
df_tem.to_csv("/Users/yslee/PycharmProjects/OPTIMAL/Results/Zone_{}.csv".format(num_ind))

tt = []
for i in range(len(df_tem.index.tolist())):
    re = df_tem.index.tolist()[i]
    tt.append(str(re)[8:16])

"""Plot"""
import matplotlib.pyplot as plt
# Font
plt.rcParams["font.family"] = "Times New Roman"
fig = plt.figure(figsize=(20, 10))
ax1 = fig.add_subplot(3, 1, 1)
ax2 = fig.add_subplot(3, 1, 2)
ax3 = fig.add_subplot(3, 1, 3)

zone_temp = df_tem.columns[df_tem.columns.str.contains("room_temp")].tolist()[0]
ax1.plot(tt, df_tem[zone_temp].tolist(), linestyle='-', color='r', drawstyle='steps-post')

set_temp = df_tem.columns[df_tem.columns.str.contains("set_temp")].tolist()[0]
ax1.plot(tt, df_tem[set_temp].tolist(), linestyle='--', color='k', drawstyle='steps-post')

ax2.plot(tt, df_tem[power].tolist(), linestyle='-', color='b', drawstyle='steps-post')
out_temp = df_tem.columns[df_tem.columns.str.contains("outdoor_temp")].tolist()[0]
ax3.plot(tt, df_tem[out_temp].tolist(), linestyle='-', color='g', drawstyle='steps-post')

gap = 210
days_num = 1
ax1.set_xticks([tt[i] for i in range(len(tt)) if i % gap * int(days_num) == 0 or tt[i] == tt[-1]])
ax2.set_xticks([tt[i] for i in range(len(tt)) if i % gap * int(days_num) == 0 or tt[i] == tt[-1]])
ax3.set_xticks([tt[i] for i in range(len(tt)) if i % gap * int(days_num) == 0 or tt[i] == tt[-1]])

ax1.set_ylabel('Temp [C]', fontsize=24)
ax2.set_ylabel('Power signal [C]', fontsize=24)
ax3.set_ylabel('Outdoor Temp [C]', fontsize=24)
ax3.set_xlabel('Time', fontsize=24)

ax1.set_yticks([0, 10, 20,25, 30, 35])
ax2.set_yticks([0, 1])
ax3.set_yticks([-5, 0, 5, 10])

ax1.set_ylim([10, 40])
ax2.set_ylim([0, 1.5])


ax1.tick_params(axis="x", labelsize=22)
ax1.tick_params(axis="y", labelsize=22)
ax2.tick_params(axis="x", labelsize=22)
ax2.tick_params(axis="y", labelsize=22)
ax3.tick_params(axis="x", labelsize=22)
ax3.tick_params(axis="y", labelsize=22)

ax1.autoscale(enable=True, axis='x', tight=True)
# ax1.autoscale(enable=True, axis='y', tight=True)
ax2.autoscale(enable=True, axis='x', tight=True)
# ax2.autoscale(enable=True, axis='y', tight=True)
ax3.autoscale(enable=True, axis='x', tight=True)
# ax3.autoscale(enable=True, axis='y', tight=True)
ax1.set_title("Indoor profile : {}".format(num_ind), fontsize=24)

ax1.grid()
ax2.grid()
ax3.grid()

plt.tight_layout()
plt.savefig("/Users/yslee/PycharmProjects/OPTIMAL/Results/profile_{}.png".format(num_ind))
plt.close()
