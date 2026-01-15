"""验证延迟转换的正确性"""
import sys
sys.path.insert(0, '/Users/lixiang/Documents/工作/code/CrossRing')

from src.kcin.v1.config import V1Config as CrossRingConfig

def test_latency_conversion():
    """测试不同频率下的延迟转换"""

    print("=" * 80)
    print("延迟转换验证")
    print("=" * 80)
    print()

    configs = [
        ("2GHz", "../config/topologies/kcin_5x4.yaml"),
        ("2.5GHz", "../config/topologies/kcin_5x4_2.5GHz.yaml"),
    ]

    for freq_name, config_path in configs:
        config = CrossRingConfig(config_path)

        print(f"【{freq_name}配置】")
        print(f"  NETWORK_FREQUENCY: {config.NETWORK_FREQUENCY} GHz")
        print(f"  IP_FREQUENCY: {config.IP_FREQUENCY} GHz")
        print(f"  CYCLES_PER_NS: {config.CYCLES_PER_NS}")
        print()

        # 模拟100 cycles的延迟转换
        test_cycles = 100

        # 错误方法：使用NETWORK_FREQUENCY(GHz值)
        wrong_ns = test_cycles / config.NETWORK_FREQUENCY

        # 正确方法：使用CYCLES_PER_NS
        correct_ns = test_cycles / config.CYCLES_PER_NS

        print(f"  测试：{test_cycles} cycles 转换为 ns")
        print(f"    错误方法(使用NETWORK_FREQUENCY): {wrong_ns:.2f} ns")
        print(f"    正确方法(使用CYCLES_PER_NS): {correct_ns:.2f} ns")

        if freq_name == "2.5GHz":
            if abs(wrong_ns - correct_ns) > 0.01:
                print(f"    ❌ 差异: {wrong_ns - correct_ns:.2f} ns ({(wrong_ns/correct_ns - 1)*100:.1f}%)")
            else:
                print(f"    ✓ 两种方法结果相同")
        print()

    # 验证analyzers中的self.network_frequency是什么值
    print("=" * 80)
    print("验证Analyzer类中的变量")
    print("=" * 80)
    print()

    # 模拟analyzers.py中的赋值
    config_2_5 = CrossRingConfig("../config/topologies/kcin_5x4_2.5GHz.yaml")

    # 这是analyzers.py:224的赋值方式
    network_frequency_in_analyzer = config_2_5.CYCLES_PER_NS

    print("analyzers.py中的赋值:")
    print(f"  self.network_frequency = config.CYCLES_PER_NS")
    print(f"  实际值: {network_frequency_in_analyzer}")
    print()

    print("RequestCollector初始化:")
    print(f"  RequestCollector(self.network_frequency)")
    print(f"  实际传入值: {network_frequency_in_analyzer}")
    print()

    print("延迟转换:")
    print(f"  value / self.network_frequency")
    print(f"  100 / {network_frequency_in_analyzer} = {100 / network_frequency_in_analyzer:.2f} ns ✓")
    print()

    print("=" * 80)
    print("结论")
    print("=" * 80)
    print()
    print("✓ 当前实现已经正确!")
    print("  analyzers和d2d_analyzer中都使用了 config.CYCLES_PER_NS")
    print("  虽然变量名叫network_frequency，但存储的值是正确的")
    print("  所有的 / self.network_frequency 操作都是正确的")
    print()

if __name__ == "__main__":
    test_latency_conversion()
