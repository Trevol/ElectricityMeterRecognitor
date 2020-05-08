# that's it!  Now, you can create a Bunch
# whenever you want to group a few variables:
from utils.bunch import Bunch

x = 40
y = 20
point = Bunch(datum=y, squared=y * y, coord=x)

# and of course you can read/write the named
# attributes you just created, add others, del
# some of them, etc, etc:
threshold = 100
if point.squared > threshold:
    point.isok = 1
print(point, point.isok)

point.isok = 2
print(point, point.isok)

point2 = Bunch(datum=y, squared=y * y, coord=50)
print(point2)
print(point)
