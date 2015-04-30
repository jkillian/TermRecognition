import math

def dtw(list1, list2, distance_func, w=None):
   w = w or float('inf')
   n = len(list1)
   m = len(list2)

   data = []
   for e in range(0, n+1):
      data.append([float('inf')] * (m+1))

   data[0][0] = 0

   w = max(w, abs(n - m))

   for i in range(1, n + 1):
      for j in range(max(1, i-w), min(m, i+w)+1):
         ele1 = list1[i-1]
         ele2 = list2[j-1]
         cost = distance_func(ele1, ele2)
         data[i][j] = cost + min(data[i-1][j], data[i][j-1], data[i-1][j-1])

   # print(data[n])
   return data[n][m]



def list_distance(list1, list2):
   return math.sqrt(sum(map(lambda x: (x[0]-x[1])**2, zip(list1, list2))))