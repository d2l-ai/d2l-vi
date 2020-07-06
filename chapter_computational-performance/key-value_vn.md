<!--
# Distributed Key-Value Store
-->

# *dịch tiêu đề phía trên*
:label:`sec_key_value`


<!--
KVStore is a place for data sharing. Think of it as a single object shared across different devices (GPUs and computers), where each device can push data in and pull data out.
-->

*dịch đoạn phía trên*

<!--
## Initialization
-->

## *dịch tiêu đề phía trên*


<!--
Let’s consider a simple example: initializing a (int, NDArray) pair into the store, and then pulling the value out:
-->

*dịch đoạn phía trên*


```{.python .input  n=1}
from d2l import mxnet as d2l
from mxnet import np, npx, kv
npx.set_np()
```

```{.python .input  n=2}
np.ones((2,3))
```

```{.python .input  n=11}
help(kv)
```

```{.python .input  n=3}
kv = kv.create('local')  # Create a local kv store.
shape = (2,3)
kv.init(3, np.ones(shape) * 2)
a = np.zeros(shape)
kv.pull(3, out = a)
print(a)
```

<!--
## Push, Aggregate, and Update
-->

## *dịch tiêu đề phía trên*


<!--
For any key that has been initialized, you can push a new value with the same shape to the key:
-->

*dịch đoạn phía trên*



```{.python .input  n=4}
kv.push(3, np.ones(shape)*8)
kv.pull(3, out = a)  # Pull out the value
print(a.asnumpy())
```


<!--
The data for pushing can be stored on any device.
Furthermore, you can push multiple values into the same key, where KVStore will first sum all of these values and then push the aggregated value.
Here we will just demonstrate pushing a list of values on CPU.
Please note summation only happens if the value list is longer than one
-->

*dịch đoạn phía trên*


```{.python .input  n=5}
contexts = [npx.cpu(i) for i in range(4)]
b = [np.ones(shape, ctx=ctx) for ctx in contexts]
kv.push(3, b)
kv.pull(3, out = a)
print(a)
```


<!--
For each push, KVStore combines the pushed value with the value stored using an updater.
The default updater is ASSIGN. You can replace the default to control how data is merged:
-->

*dịch đoạn phía trên*



```{.python .input  n=6}
def update(key, input, stored):
    print(f'update on key: {key}')
    stored += input * 2
kv._set_updater(update)
kv.pull(3, out=a)
print(a)
```

```{.python .input  n=7}
kv.push(3, np.ones(shape))
kv.pull(3, out=a)
print(a)
```

<!--
## Pull
-->

## *dịch tiêu đề phía trên*


<!--
You’ve already seen how to pull a single key-value pair. Similarly, to push, you can pull the value onto several devices with a single call:
-->

*dịch đoạn phía trên*



```{.python .input  n=8}
b = [np.ones(shape, ctx=ctx) for ctx in contexts]
kv.pull(3, out = b)
print(b[1])
```


<!--
## Handle a List of Key-Value Pairs
-->

## *dịch tiêu đề phía trên*


<!--
All operations introduced so far involve a single key. KVStore also provides an interface for a list of key-value pairs.
-->

*dịch đoạn phía trên*


<!--
For a single device:
-->

*dịch đoạn phía trên*


```{.python .input  n=9}
keys = [5, 7, 9]
kv.init(keys, [np.ones(shape)]*len(keys))
kv.push(keys, [np.ones(shape)]*len(keys))
b = [np.zeros(shape)]*len(keys)
kv.pull(keys, out = b)
print(b[1])
```


<!--
For multiple devices:
-->

*dịch đoạn phía trên*


```{.python .input  n=10}
b = [[np.ones(shape, ctx=ctx) for ctx in contexts]] * len(keys)
kv.push(keys, b)
kv.pull(keys, out = b)
print(b[1][1])
```

## Những người thực hiện
Bản dịch trong trang này được thực hiện bởi:
<!--
Tác giả của mỗi Pull Request điền tên mình và tên những người review mà bạn thấy
hữu ích vào từng phần tương ứng. Mỗi dòng một tên, bắt đầu bằng dấu `*`.
Tên đầy đủ của các reviewer có thể được tìm thấy tại https://github.com/aivivn/d2l-vn/blob/master/docs/contributors_info.md
-->

* Đoàn Võ Duy Thanh
<!-- Phần 1 -->
* 

<!-- Phần 2 -->
* 

<!-- Phần 3 -->
* 

<!-- Phần 4 -->
* 

<!-- Phần 5 -->
* 

<!-- Phần 6 -->
* 