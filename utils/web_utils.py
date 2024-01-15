def make_common_response(state: int, message: str, data):
    """
    生成统一响应对象
    :param state: 响应状态码
    :param message: 响应消息
    :param data: payload 携带数据信息
    :return: 统一响应体
    """
    return {
        'code': state,
        'message': message,
        'data': data
    }


def make_common_page_result(data: list, total: int, page: int):
    """
    返回统一分页数据信息
    :param data: 数据列表
    :param total: 总数
    :param page: 页码
    :return: 统一分页
    """
    return {
        'dataList': data,
        'total': total,
        'page': page
    }
