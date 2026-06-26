from pymilvus import MilvusClient,DataType
import sys
import os
# 修复 milvus-lite 在 Windows 平台上 os.rename 无法覆盖已有文件的系统级 Bug
if sys.platform == "win32":
    os.rename = os.replace

client = MilvusClient("milvus_demo.db")
schema = MilvusClient.create_schema()
# 添加字段时，可以通过将 is_primary 属性设置为 True 来明确说明该字段是主字段。主字段默认接受Int64值。在这种情况下，主字段值应为整数，类似于12345 。如果选择在主字段中使用VarChar值，则其值应为字符串，类似于my_entity_1234 。
# 建议您在所有情况下都使用 autoId ，除非手动设置主键是有益的。
schema.add_field(
    field_name="my_id",
    datatype=DataType.INT64,
    is_primary=True,
    auto_id=False,
)
# FLOAT_VECTOR 值表示该向量场持有 32 位浮点数列表
schema.add_field(
    field_name="my_vector",
    datatype=DataType.FLOAT_VECTOR,
    # highlight-next-line
    dim=5
)
# Milvus 支持多种标量字段类型，包括VarChar、Boolean、Int、Float 和Double。
schema.add_field(
    field_name="my_varchar",
    datatype=DataType.VARCHAR,
    # highlight-next-line
    max_length=512
)
# Milvus 支持的数字类型有 Int8,
# Int16,
# Int32,
# Int64,
# Float 和 Double 。
schema.add_field(
    field_name="my_int64",
    datatype=DataType.INT64,
)
schema.add_field(
    field_name="my_bool",
    datatype=DataType.BOOL,
)
# 添加json字段
schema.add_field(
    field_name="my_json",
    datatype=DataType.JSON,
)
# 数组字段 数组字段中所有元素的数据类型应相同。
schema.add_field(
    field_name="my_array",
    datatype=DataType.ARRAY,
    element_type=DataType.VARCHAR,
    max_capacity=5,
    max_length=512,
)
# Create the collection
if client.has_collection("demo_autoid"):
    client.drop_collection("demo_autoid")
client.create_collection(collection_name="demo_autoid", schema=schema)

