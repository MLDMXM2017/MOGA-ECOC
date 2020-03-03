import xlrd
from xlutils.copy import copy 

     
def logExcel(name,results,excelname):
    rexcel=xlrd.open_workbook("records_NSGA/"+excelname+".xls")
    col=rexcel.sheets()[0].ncols
    row=1
    excel=copy(rexcel)
    tabel=excel.get_sheet(0)
    tabel.write(0,col,name)
    for i in range(len(results)):
        tabel.write(row,col,results[i])
        row=row+1
    excel.save("records_NSGA/"+excelname+".xls")
    
    