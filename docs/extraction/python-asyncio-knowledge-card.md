# Python asyncio 知识卡

## 它是什么

`asyncio` 是 Python 自带的异步编程库。它特别适合网络请求、文件读写等“发出请求后需要等待”的
场景。等待模型 API 返回时，程序不必用额外线程；它把等待期间的任务交给事件循环管理。

本项目的 DeepSeek 图谱抽取要等待 HTTP 响应，因此节点演示使用 `asyncio`。

## 三个关键字

```python
async def main():
    graph = await _extract_graph(...)

asyncio.run(main())
```

- `async def`：定义异步函数。调用它不会立刻执行全部逻辑，而是得到一个待运行任务。
- `await`：暂停当前异步函数，等待另一个异步操作完成。这里等待模型返回节点。
- `asyncio.run(...)`：创建事件循环，运行异步函数，结束后关闭事件循环。

事件循环可以理解为异步任务的调度器：它负责在“请求模型”“等待网络”“关闭连接”等任务之间安排
执行时机。

## 本次节点演示中的数据流

```text
asyncio.run(main())
  -> 创建一个事件循环
  -> await _extract_graph(...)：发送 DeepSeek 节点抽取请求并等待响应
  -> await client.aclose()：关闭 HTTP 连接
  -> 事件循环结束
```

`_extract_graph()` 和 `client.aclose()` 都是异步操作，必须在同一个事件循环中完成。

## 这次遇到的错误

错误写法：

```python
graph = asyncio.run(_extract_graph(...))
asyncio.run(client.aclose())
```

这会创建两个事件循环。HTTP 连接在第一个循环中建立，第一个循环结束时已经关闭；第二个循环再试图
关闭同一连接，会出现：

```text
RuntimeError: Event loop is closed
```

正确写法：

```python
async def main() -> None:
    client = create_deepseek_graph_builder()
    try:
        graph = await _extract_graph(...)
    finally:
        await client.aclose()


if __name__ == "__main__":
    asyncio.run(main())
```

`finally` 的作用是：即使模型请求失败，也仍然关闭网络连接。

## 什么时候需要 asyncio

需要：调用异步 HTTP 客户端、并发请求多个模型、异步数据库或文件操作。

不需要：普通计算、简单字符串处理、同步函数。不要为了“看起来高级”而给每个函数加 `async`；只有函数
内部确实要 `await` 异步操作时才使用。

## 在本项目中的位置

- 节点演示入口：[contract.py](../../src/medical_kg_sourceprep/extraction/graph_builder/contract.py)
- 模型请求封装：[schema.py](../../src/medical_kg_sourceprep/extraction/graph_builder/schema.py)
- HTTP 客户端关闭逻辑：[client.py](../../src/medical_kg_sourceprep/extraction/graph_builder/client.py)
