import matplotlib.pyplot as plt
import numpy as np
from prettytable import PrettyTable
x = PrettyTable()
file = open('../dataset/people.txt','r')

num = []
for line in file:
    line = line.rstrip()
    words = line.split()
    if len(words) != 2 :
        continue
    num.append(int(words[1]))

num = np.asarray(num)

num_list = []
num_label = []
temp = np.sum(num == 1)
num_list.append(temp)
num_label.append('only 1')

for i in range(12,23,10):
    temp = np.sum(num<i)
    temp_1 = np.sum(num <=i-10)
    num_list.append(temp-temp_1)
    num_label.append(str(i-10)+'--'+str(i))

temp = np.sum(num >= 22)
num_list.append(temp)
num_label.append('more than 22')

plt.pie(num_list, labels = num_label)
plt.title("numbrt of photos each person have")
plt.show()
plt.savefig('../screenshot/data_pie_chart.png')

len(num_label)
x=PrettyTable(['','number','percentage %'])
for i in range(len(num_label)):
    x.add_row([num_label[i], num_list[i],100*num_list[i]/np.sum(num_list)])
x.add_row(['total',np.sum(num_list),100])

print(x)
