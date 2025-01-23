ubatchsize = 1

ubatchsize_range = [ubatchsize, ubatchsize + 1, 1]
print(ubatchsize_range)

for id in range(*ubatchsize_range):
    print(id)

from collections import OrderedDict as ODict
stats = ODict()
stats[1]=ODict()
stats[1][0]=None
print(stats)

a = [1,2,3]
print(list(a))

print(a.index(2))