import pandas as pd
import  openpyxl

matrix = [[1, 2, 3],
          [4, 5, 6],
          [7, 8, 9]]
df = pd.DataFrame(matrix)


file_path ='C:\\Users\\Shudhanshu\\PycharmProjects\\B-spline\\file.xlsx'
df.to_excel(file_path, index=False)
