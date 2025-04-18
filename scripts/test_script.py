def analyze_value_ratios(result):
    """
    统计字典中每个value对256的比值，并进行简单分析

    参数:
        result (dict): 需要分析的字典

    返回:
        dict: 包含比值和统计信息的字典
    """
    if not isinstance(result, dict):
        raise ValueError("输入必须是字典类型")

    # 计算每个value对256的比值
    ratios = {key: value / 256 for key, value in result.items()}

    # 统计信息
    if ratios:  # 确保字典不为空
        values = list(result.values())
        ratio_values = list(ratios.values())

        stats = {
            "min_ratio": min(ratio_values),
            "max_ratio": max(ratio_values),
            "avg_ratio": sum(ratio_values) / len(ratio_values),
            "min_value": min(values),
            "max_value": max(values),
            "avg_value": sum(values) / len(values),
            "total_values": len(values),
            "ratio_distribution": {"less_than_1": sum(1 for r in ratio_values if r < 1), "equal_to_1": sum(1 for r in ratio_values if r == 1), "greater_than_1": sum(1 for r in ratio_values if r > 1)},
        }
    else:
        stats = {}

    return {"ratios": ratios, "statistics": stats}


# 示例用法
if __name__ == "__main__":
    # 示例字典
    sample_dict = {"a": 128, "b": 256, "c": 512, "d": 64, "e": 1024, "f": 300}

    analysis = analyze_value_ratios(sample_dict)

    print("原始字典:", sample_dict)
    print("\n比值结果:")
    for key, ratio in analysis["ratios"].items():
        print(f"{key}: {ratio:.4f}")

    print("\n统计信息:")
    stats = analysis["statistics"]
    print(f"最小比值: {stats['min_ratio']:.4f}")
    print(f"最大比值: {stats['max_ratio']:.4f}")
    print(f"平均比值: {stats['avg_ratio']:.4f}")
    print(f"\n原始值最小值: {stats['min_value']}")
    print(f"原始值最大值: {stats['max_value']}")
    print(f"原始值平均值: {stats['avg_value']:.2f}")
    print(f"\n比值分布:")
    print(f"小于1: {stats['ratio_distribution']['less_than_1']}个")
    print(f"等于1: {stats['ratio_distribution']['equal_to_1']}个")
    print(f"大于1: {stats['ratio_distribution']['greater_than_1']}个")
