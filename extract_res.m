function res = extract_res(table)
res = sortrows(table, 1, 'descend');
assert(all(size(res) == [1, 3]), "Feret table had more than 1 row")
res = res(1,:);
