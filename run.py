from simulateflow import flow

col_number = 1
sheet_name = '病程记录_首次病程'

def cache():
    with open('./make_task/case_cache.txt', 'r', encoding='utf-8') as cc:
        case_number = cc.read()
    return case_number

row_number = int(cache())

while row_number <= 1300:
    # row_number = int(cache())
    row_number += 1
    flow(sheet_name, row_number, col_number)
    with open('./make_task/case_cache.txt', 'w', encoding='utf-8') as cc:
        cc.write(str(row_number))
