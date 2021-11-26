def test_list_pre():
    prepare_list = locals()
    for i in range(16):
        prepare_list['list_' + str(i)] = []
        prepare_list['list_' + str(i)].append(('我是第' + str(i)) + '个list')


def get_variable_name(variable):
    callers_local_vars = inspect.currentframe().f_back.f_locals.items()
    return [var_name for var_name, var_val in callers_local_vars if var_val is variable]


if __name__ == '__main__':
    prepare_list = locals()
    for i in range(16):
        prepare_list['list_' + str(i)] = []
        prepare_list['list_' + str(i)].append(('我是第' + str(i)) + '个list')
    a = get_variable_name(prepare_list['list_0']).pop()
    b = get_variable_name(prepare_list['list_1']).pop()
    print(a)
    print(b)


if __name__ == '__main__':
    test_list_pre()