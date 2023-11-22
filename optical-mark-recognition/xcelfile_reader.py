import xlrd
d = {}
wb = xlrd.open_workbook('sample.xlsx')
sh = wb.sheet_by_index(0)
for i in range(3):
    cell_value_class = int(sh.cell(i, 0).value)
    cell_value_id = int(sh.cell(i, 1).value)
    d[cell_value_class] = cell_value_id
print(d)
