mean = 1.8
li = 1
ri = 2
lw = (ri - mean) / (ri - li)
rw = (mean - li) / (ri - li)
mean_cal = lw * li + rw * ri
print(mean_cal)