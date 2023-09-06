import copy
import numpy as np


def single_elements_filter(input, length=1, replace_element=0):
    # 删除连续出现小于等于length的部分元素
    # 使用 replace_element 替换被过滤的元素

    output = copy.deepcopy(input)
    elements_start_duration = []
    s = 0
    for i in range(output.shape[0]):
        if i == s:
            continue
        if len(set(output[s:i + 1])) == 1:
            continue

        elements_start_duration.append(
            (output[s].item(), s, i - s)
        )
        s = i
    elements_start_duration.append(
        (output[s].item(), s, i - s + 1)
    )

    for esd in elements_start_duration:
        if esd[2] <= length:
            output[esd[1]:esd[1] + esd[2]] = replace_element

    return output


if __name__ == '__main__':
    example = np.array(
        [0, 0, 0, 0, 1, 1, 2, 3, 3, 3, 0, 0, 1, 1]
    )

    print(single_elements_filter(example, 2))
