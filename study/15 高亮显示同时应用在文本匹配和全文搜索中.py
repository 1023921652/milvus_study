# 本例展示了如何高亮显示TEXT_MATCH 过滤器匹配的术语。
#
# BM25 全文搜索使用"test" 作为查询词
#
# queries 参数将"my doc" 添加到高亮列表中
#
# 高亮显示器将所有匹配词（"my",
# "test",
# "doc" ）与{ 和 } 包在一起
highlighter = LexicalHighlighter(
    pre_tags=["{"],
    post_tags=["}"],
    highlight_search_text=True,   # Also highlight BM25 term
    highlight_query=[                     # Additional TEXT_MATCH terms to highlight
        {"type": "TextMatch", "field": "text", "text": "my doc"},
    ],
)

results = client.search(
    collection_name=COLLECTION_NAME,
    data=["test"],
    anns_field="sparse_vector",
    limit=10,
    search_params=SEARCH_PARAMS,
    output_fields=["text"],
    # highlight-next-line
    highlighter=highlighter,
)

for hit in results[0]:
    print(f"  {hit.get('highlight', {}).get('text', [])}")
print()
