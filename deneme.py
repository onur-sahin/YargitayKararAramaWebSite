
y = "2021-01-11 00:00:00"
from datetime import date, datetime
x = datetime(2022,5,6)

y = datetime.strptime(y, "%Y-%m-%d %H:%M:%S")

print(y.strftime("%Y/%m/%d"))
