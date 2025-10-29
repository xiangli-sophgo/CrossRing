"""
对比不同ORDERING_GRANULARITY设置下的仿真结果差异
帮助诊断为什么两个层级的结果看起来一样
"""
import sys
import io
from pathlib import Path

# 设置UTF-8编码输出
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def analyze_why_same_results():
    """分析为什么两种粒度可能产生相同结果"""
    print("=" * 80)
    print("分析：为什么两种粒度的结果可能看起来一样")
    print("=" * 80)

    print("\n可能的原因：")
    print("\n1. 测试流量特征")
    print("   - 如果每个节点只有一个IP发送流量，IP层级=节点层级")
    print("   - 需要同一节点多个IP同时发送流量才能看出差异")

    print("\n2. 结果观察维度")
    print("   - 如果只看节点级别的延迟/带宽，两种粒度结果相同")
    print("   - 需要看IP级别的保序行为才能看出差异")

    print("\n3. 保序冲突场景")
    print("   - 只有当多个IP的flit在同一下环点竞争时才有差异")
    print("   - IP层级：不同IP独立保序，可能交错下环")
    print("   - 节点层级：同节点所有IP共享保序，严格顺序下环")

    print("\n4. 示例场景说明")
    print("   假设节点0有gdma_0和gdma_1两个IP，都向节点4发送flit")
    print("   时刻T，两个flit同时到达下环点：")
    print("     - gdma_0的flit: order_id=5")
    print("     - gdma_1的flit: order_id=3")
    print("")
    print("   IP层级 (granularity=0):")
    print("     - gdma_0的key: (0, gdma_0, 4, ddr_0, TL)")
    print("     - gdma_1的key: (0, gdma_1, 4, ddr_0, TL)")
    print("     - 两个key不同，独立保序")
    print("     - 如果gdma_0期望order_id=5, gdma_1期望order_id=3")
    print("     - 两个flit都可以下环（各自序列正确）")
    print("")
    print("   节点层级 (granularity=1):")
    print("     - gdma_0的key: (0, 4, TL)")
    print("     - gdma_1的key: (0, 4, TL)")
    print("     - 两个key相同，共享保序")
    print("     - 如果当前期望order_id=3")
    print("     - gdma_1可以下环，gdma_0必须等待")


def create_test_scenario():
    """创建能够明显区分两种粒度的测试场景"""
    print("\n" + "=" * 80)
    print("能够区分两种粒度的测试场景")
    print("=" * 80)

    print("\n【场景设计】")
    print("节点0有两个IP: gdma_0和gdma_1")
    print("目标：同时向节点4发送flit")
    print("")
    print("发送顺序（上环顺序）：")
    print("  T1: gdma_0发送flit1 (会得到order_id)")
    print("  T2: gdma_1发送flit2 (会得到order_id)")
    print("  T3: gdma_0发送flit3 (会得到order_id)")
    print("  T4: gdma_1发送flit4 (会得到order_id)")
    print("")
    print("假设flit1先到达下环点，但flit2还在路上")
    print("")

    from src.utils.components.flit import Flit

    # 模拟IP层级
    print("\n【IP层级 (ORDERING_GRANULARITY=0)】")
    Flit.reset_order_ids()

    flits_ip = []
    for i, (src_type, desc) in enumerate([
        ("gdma_0", "flit1"), ("gdma_1", "flit2"),
        ("gdma_0", "flit3"), ("gdma_1", "flit4")
    ]):
        order_id = Flit.get_next_order_id(
            src_node=0, src_type=src_type,
            dest_node=4, dest_type="ddr_0",
            packet_category="REQ", granularity=0
        )
        flits_ip.append((src_type, desc, order_id))
        print(f"  {desc} ({src_type}): order_id={order_id}")

    print("\n  保序检查逻辑：")
    print("  - gdma_0的key: (0, gdma_0, 4, ddr_0, TL)")
    print("  - gdma_1的key: (0, gdma_1, 4, ddr_0, TL)")
    print("  - 两个key独立，期望order_id分别维护")
    print("  - gdma_0期望: 1,2,3... | gdma_1期望: 1,2,3...")
    print("  - flit1可下环(gdma_0期望1)，flit2可下环(gdma_1期望1)")
    print("  - 即使flit3先到，如果flit1还没下环，flit3也不能下环")

    # 模拟节点层级
    print("\n【节点层级 (ORDERING_GRANULARITY=1)】")
    Flit.reset_order_ids()

    flits_node = []
    for i, (src_type, desc) in enumerate([
        ("gdma_0", "flit1"), ("gdma_1", "flit2"),
        ("gdma_0", "flit3"), ("gdma_1", "flit4")
    ]):
        order_id = Flit.get_next_order_id(
            src_node=0, src_type=src_type,
            dest_node=4, dest_type="ddr_0",
            packet_category="REQ", granularity=1
        )
        flits_node.append((src_type, desc, order_id))
        print(f"  {desc} ({src_type}): order_id={order_id}")

    print("\n  保序检查逻辑：")
    print("  - gdma_0的key: (0, 4, TL)")
    print("  - gdma_1的key: (0, 4, TL)")
    print("  - 两个key相同，共享期望order_id")
    print("  - 期望顺序: 1,2,3,4...")
    print("  - flit1可下环(期望1)，flit2必须等flit1下环后才能下(期望2)")
    print("  - 严格按照上环顺序下环")

    print("\n【关键差异】")
    print("假设flit到达下环点的顺序是: flit1, flit3, flit2, flit4")
    print("")
    print("IP层级:")
    print("  - flit1到达: order_id=1, gdma_0期望1 ✓ 下环")
    print("  - flit3到达: order_id=2, gdma_0期望2 ✓ 下环")
    print("  - flit2到达: order_id=1, gdma_1期望1 ✓ 下环")
    print("  - flit4到达: order_id=2, gdma_1期望2 ✓ 下环")
    print("  实际下环顺序: flit1, flit3, flit2, flit4")
    print("")
    print("节点层级:")
    print("  - flit1到达: order_id=1, 期望1 ✓ 下环，期望变为2")
    print("  - flit3到达: order_id=3, 期望2 ✗ 等待")
    print("  - flit2到达: order_id=2, 期望2 ✓ 下环，期望变为3")
    print("  - flit3检查: order_id=3, 期望3 ✓ 下环，期望变为4")
    print("  - flit4到达: order_id=4, 期望4 ✓ 下环")
    print("  实际下环顺序: flit1, flit2, flit3, flit4 (严格按上环顺序)")


def check_traffic_pattern():
    """检查你的流量模式是否能体现差异"""
    print("\n" + "=" * 80)
    print("如何检查你的流量是否能体现粒度差异")
    print("=" * 80)

    print("\n【需要满足的条件】")
    print("1. 同一物理节点有多个IP同时发送流量")
    print("   - 例如：节点0同时有gdma_0和gdma_1发送")
    print("   - 如果每个节点只有一个活跃IP，两种粒度等价")

    print("\n2. 这些IP的flit会在环上相遇/乱序")
    print("   - 不同IP的flit经过不同路径，到达下环点顺序打乱")
    print("   - 如果流量很稀疏，flit之间不干扰，看不出差异")

    print("\n3. 观察IP级别的保序行为")
    print("   - 需要trace每个flit的下环时间")
    print("   - 比较同节点不同IP的flit下环顺序")

    print("\n【如何验证配置是否生效】")
    print("方法1: 查看日志中的order_id分配")
    print("  - 同节点不同IP的flit，IP层级order_id独立从1开始")
    print("  - 同节点不同IP的flit，节点层级order_id连续递增")

    print("\n方法2: 查看保序等待事件")
    print("  - 节点层级应该有更多保序等待（因为更严格）")
    print("  - 统计每个下环点的保序阻塞次数")

    print("\n方法3: 对比平均延迟")
    print("  - 节点层级可能延迟略高（更多等待）")
    print("  - 但差异可能很小，取决于流量模式")


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("ORDERING_GRANULARITY效果诊断与对比")
    print("=" * 80)

    try:
        analyze_why_same_results()
        create_test_scenario()
        check_traffic_pattern()

        print("\n" + "=" * 80)
        print("总结")
        print("=" * 80)
        print("\n如果你的结果两个层级看起来一样，最可能的原因是：")
        print("  1. 流量模式不够复杂，无法体现差异")
        print("  2. 观察的指标（如平均延迟）对粒度不敏感")
        print("  3. 同一节点的不同IP流量没有时间重叠")
        print("\n要验证参数是否生效：")
        print("  1. 在日志中查找order_id分配记录")
        print("  2. 统计保序等待事件的key")
        print("  3. 创建专门的测试场景（多IP并发）")

    except Exception as e:
        print(f"\n❌ 失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
