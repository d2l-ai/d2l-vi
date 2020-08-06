<!--
# Compilers and Interpreters
-->

# Trình biên dịch và Trình thông dịch
:label:`sec_hybridize`

<!--
So far, this book has focused on imperative programming, which makes use of statements such as `print`, `+` or `if` to change a program’s state.
Consider the following example of a simple imperative program.
-->

Cho đến nay, ta mới chỉ tập trung vào lập trình mệnh lệnh, kiểu lập trình sử dụng các câu lệnh như `print`, `+` hay `if` để thay đổi trạng thái của chương trình.
Hãy cùng xét ví dụ đơn giản sau về lập trình mệnh lệnh.

```{.python .input  n=1}
def add(a, b):
    return a + b

def fancy_func(a, b, c, d):
    e = add(a, b)
    f = add(c, d)
    g = add(e, f)
    return g

print(fancy_func(1, 2, 3, 4))
```

<!--
Python is an interpreted language. 
When evaluating `fancy_func` it performs the operations making up the function's body *in sequence*. 
That is, it will evaluate `e = add(a, b)` and it will store the results as variable `e`, thereby changing the program's state. 
The next two statements `f = add(c, d)` and `g = add(e, f)` will be excecuted similarly, performing additions and storing the results as variables. 
:numref:`fig_compute_graph` illustrates the flow of data.
-->

Python là một ngôn ngữ thông dịch.
Khi thực hiện hàm `fancy_func`, nó thực thi các lệnh trong thân hàm một cách *tuần tự*.
Như vậy, nó sẽ chạy lệnh `e = add(a, b)` rồi sau đó lưu kết quả vào biến `e`, làm cho trạng thái chương trình thay đổi.
Hai câu lệnh tiếp theo `f = add(c, d)` và `g = add(e, f)` sẽ được thực thi tương tự, thực hiện phép cộng và lưu kết quả vào các biến.
:numref:`fig_compute_graph` sẽ minh họa luồng dữ liệu.

<!--
![Data flow in an imperative program.](../img/computegraph.svg)
-->

![Luồng dữ liệu trong lập trình mệnh lệnh.](../img/computegraph.svg)
:label:`fig_compute_graph`

<!--
Although imperative programming is convenient, it may be inefficient. 
On one hand, even if the `add` function is repeatedly called throughout `fancy_func`, Python will execute the three function calls individually. 
If these are executed, say, on a GPU (or even on multiple GPUs), the overhead arising from the Python interpreter can become overwhelming. 
Moreover, it will need to save the variable values of `e` and `f` until all the statements in `fancy_func` have been executed. 
This is because we do not know whether the variables `e` and `f` will be used by other parts of the program after the statements `e = add(a, b)` and `f = add(c, d)` have been executed.
-->

Mặc dù lập trình mệnh lệnh rất thuận tiện, nhưng nó lại không quá hiệu quả. 
Ở đây nếu hàm `add` được gọi nhiều lần trong `fancy_func`, Python cũng sẽ thực thi ba lần gọi hàm độc lập.
Nếu điều này xảy ra, giả sử trên một GPU (hay thậm chí nhiều GPU), chi phí phát sinh từ trình thông dịch Python có thể sẽ rất lớn.
Hơn nữa, nó sẽ cần phải lưu giá trị các biến `e` và `f` cho tới khi tất cả các lệnh trong `fancy_func` thực thi xong.
Điều này là do ta không biết liệu biến `e` và `f` có được sử dụng bởi các phần chương trình khác sau hai lệnh `e = add(a, b)` và `f = add(c, d)` nữa hay không.



<!--
## Symbolic Programming
-->

## Lập trình Ký hiệu

<!--
Consider the alternative, symbolic programming where computation is usually performed only once the process has been fully defined. 
This strategy is used by multiple deep learning frameworks, including Theano, Keras and TensorFlow (the latter two have since acquired imperative extensions). 
It usually involves the following steps:
-->

Lập trình ký hiệu là kiểu lập trình mà ở đó các tính toán thường chỉ được thực hiện một khi chương trình đã được định nghĩa đầy đủ.
Cơ chế này được sử dụng trong nhiều framework, bao gồm: Theano, Keras và TensorFlow (hai framework sau đã hỗ trợ lập trình mệnh lệnh).
Lập trình ký hiệu thường gồm những bước sau:

<!--
1. Define the operations to be executed.
2. Compile the operations into an executable program.
3. Provide the required inputs and call the compiled program for execution.
-->

1. Khai báo các thao tác sẽ được thực thi.
2. Biên dịch các thao tác thành chương trình có thể chạy được.
3. Thực thi bằng cách cung cấp đầu vào và gọi chương trình đã được biên dịch.

<!--
This allows for a significant amount of optimization. 
First off, we can skip the Python interpreter in many cases, thus removing a performance bottleneck that can become significant on multiple fast GPUs paired with a single Python thread on a CPU. 
Secondly, a compiler might optimize and rewrite the above code into `print((1 + 2) + (3 + 4))` or even `print(10)`. 
This is possible since a compiler gets to see the full code before turning it into machine instructions. 
For instance, it can release memory (or never allocate it) whenever a variable is no longer needed. 
Or it can transform the code entirely into an equivalent piece. 
To get a better idea consider the following simulation of imperative programming (it is Python after all) below.
-->

Quy trình trên cho phép chúng ta tối ưu hóa chương trình một cách đáng kể.
Đầu tiên, ta có thể bỏ qua trình thông dịch Python trong nhiều trường hợp, từ đó loại bỏ được vấn đề nghẽn cổ chai có thể ảnh hưởng nghiêm trọng tới tốc độ tính toán khi sử dụng nhiều GPU tốc độ cao với một luồng Python duy nhất trên CPU. 
Thứ hai, trình biên dịch có thể tối ưu và viết lại mã nguồn thành `print((1 + 2) + (3 + 4))` hoặc thậm chí `print(10)`.
Điều này hoàn toàn khả thi bởi trình biên dịch có thể thấy toàn bộ mã nguồn rồi mới dịch sang mã máy.
Ví dụ, nó có thể giải phóng bộ nhớ (hoặc không cấp phát) bất cứ khi nào một biến không còn được dùng đến.
Hoặc nó có thể chuyển toàn bộ mã nguồn thành một đoạn tương đương.
Để hiểu rõ hơn vấn đề, dưới đây ta sẽ thử mô phỏng quá trình lập trình mệnh lệnh (dựa trên Python). 


```{.python .input  n=2}
def add_():
    return '''
def add(a, b):
    return a + b
'''

def fancy_func_():
    return '''
def fancy_func(a, b, c, d):
    e = add(a, b)
    f = add(c, d)
    g = add(e, f)
    return g
'''

def evoke_():
    return add_() + fancy_func_() + 'print(fancy_func(1, 2, 3, 4))'

prog = evoke_()
print(prog)
y = compile(prog, '', 'exec')
exec(y)
```


<!--
The differences between imperative (interpreted) programming and symbolic programming are as follows:
-->

Sự khác biệt giữa lập trình mệnh lệnh (thông dịch) và lập trình ký hiệu như sau:

<!--
* Imperative programming is easier. 
When imperative programming is used in Python, the majority of the code is straightforward and easy to write. 
It is also easier to debug imperative programming code. 
This is because it is easier to obtain and print all relevant intermediate variable values, or use Python’s built-in debugging tools.
* Symbolic programming is more efficient and easier to port. 
It makes it easier to optimize the code during compilation, while also having the ability to port the program into a format independent of Python. 
This allows the program to be run in a non-Python environment, thus avoiding any potential performance issues related to the Python interpreter.
-->

* Lập trình mệnh lệnh dễ hơn.
Khi lập trình mệnh lệnh được sử dụng trong Python, mã nguồn trông rất trực quan và dễ viết.
Mã nguồn của lập trình mệnh lệnh cũng dễ gỡ lỗi hơn.
Điều này là do ta có thể dễ dàng lấy và in ra giá trị của các biến trung gian liên quan, hoặc sử dụng công cụ gỡ lỗi có sẵn của Python.
* Lập trình ký hiệu lại hiệu quả hơn và dễ sử dụng trên nền tảng khác. 
Nó giúp việc tối ưu mã nguồn trong quá trình biên dịch trở nên dễ dàng hơn, đồng thời cho phép ta chuyển đổi chương trình sang một định dạng khác không phụ thuộc vào Python.
Do đó chương trình có thể chạy trong các môi trường khác ngoài Python, từ đó tránh được mọi vấn đề tiềm ẩn về hiệu năng liên quan tới trình thông dịch Python.


<!--
## Hybrid Programming
-->

## Lập trình Hybrid

<!--
Historically most deep learning frameworks choose between an imperative or a symbolic approach. 
For example, Theano, TensorFlow (inspired by the latter), Keras and CNTK formulate models symbolically. 
Conversely Chainer and PyTorch take an imperative approach. 
An imperative mode was added TensorFlow 2.0 (via Eiger) and Keras in later revisions. 
When designing Gluon, developers considered whether it would be possible to combine the benefits of both programming models. 
This led to a hybrid model that lets users develop and debug using pure imperative programming, while having the ability to convert most programs into symbolic programs to be run when product-level computing performance and deployment are required.
-->

Trong quá khứ, hầu hết các framework đều chọn một trong hai phương án tiếp cận: lập trình mệnh lệnh hoặc lập trình ký hiệu.
Ví dụ như Theano, TensorFlow, Keras và CNTK đều xây dựng mô hình dạng ký hiệu. 
Ngược lại, Chainer và PyTorch tiếp cận theo hướng lập trình mệnh lệnh.
Mô hình kiểu mệnh lệnh đã được bổ sung vào TensorFlow 2.0 (thông qua chế độ Eager) và Keras trong những bản cập nhật sau này.
Khi thiết kế Gluon, các nhà phát triển đã cân nhắc liệu rằng có thể kết hợp ưu điểm của cả hai mô hình lập trình lại với nhau hay không.
Điều này đã dẫn đến mô hình hybrid, giúp người dùng phát triển và gỡ lỗi bằng lập trình mệnh lệnh thuần, 
đồng thời mang lại khả năng chuyển đổi hầu như toàn bộ chương trình sang dạng ký hiệu khi cần triển khai thành sản phẩm với hiệu năng tính toán cao. 

<!--
In practice this means that we build models using either the `HybridBlock` or the `HybridSequential` and `HybridConcurrent` classes. 
By default, they are executed in the same way `Block` or `Sequential` and `Concurrent` classes are executed in imperative programming. 
`HybridSequential` is a subclass of `HybridBlock` (just like `Sequential` subclasses `Block`). 
When the `hybridize` function is called, Gluon compiles the model into the form used in symbolic programming. 
This allows one to optimize the compute-intensive components without sacrifices in the way a model is implemented. 
We will illustrate the benefits below, focusing on sequential models and blocks only (the concurrent composition works analogously).
-->

Trong thực tiễn, ta sẽ xây dựng mô hình bằng lớp `HybridBlock` hoặc `HybridSequential` và `HybridConcurrent`.
Mặc định, chúng được thực thi giống hệt như cách lớp `Block` hoặc `Sequential` và `Concurrent` được thực thi trong kiểu lập trình mệnh lệnh.
`HybridSequential` là một lớp con của `HybridBlock` (cũng như `Sequential` là lớp con của `Block`). 
Khi hàm `hybridize` được gọi, Gluon biên dịch mô hình thành định dạng được dùng trong lập trình ký hiệu.
Điều này cho phép ta tối ưu các thành phần nặng về mặt tính toán mà không cần có nhiều thay đổi trong cách lập trình mô hình. 
Chúng tôi sẽ minh hoạ lợi ích của việc này ở ví dụ bên dưới, tập trung vào các mô hình `Sequential` và `Block` (mô hình `Concurrent` cũng sẽ hoạt động tương tự).


<!--
## HybridSequential
-->

## HybridSequential

<!--
The easiest way to get a feel for how hybridization works is to consider deep networks with multiple layers. 
Conventionally the Python interpreter will need to execute the code for all layers to generate an instruction that can then be forwarded to a CPU or a GPU. 
For a single (fast) compute device this does not cause any major issues. 
On the other hand, if we use an advanced 8-GPU server such as an AWS P3dn.24xlarge instance Python will struggle to keep all GPUs busy. 
The single-threaded Python interpreter becomes the bottleneck here. 
Let's see how we can address this for significant parts of the code by replacing `Sequential` by `HybridSequential`. We begin by defining a simple MLP.
-->

Cách đơn giản nhất để hiểu cách hoạt động của phép hybrid hóa là xem xét các mạng sâu đa tầng.
Thông thường, trình thông dịch Python sẽ thực thi mã nguồn cho tất cả các tầng để sinh một lệnh mà sau đó có thể được truyền tới CPU hoặc GPU. 
Đối với thiết bị tính toán đơn (và nhanh), quá trình trên không gây ra vấn đề lớn nào cả.
Mặt khác, nếu ta sử dụng một máy chủ tiên tiến có 8 GPU, ví dụ như P3dn.24xlarge trên AWS, Python sẽ gặp khó khăn trong việc tận dụng tất cả các GPU cùng lúc. 
Lúc này trình thông dịch Python đơn luồng sẽ trở thành nút nghẽn cổ chai.
Hãy xem làm thế nào để giải quyết vấn đề trên cho phần lớn đoạn mã nguồn bằng cách thay `Sequential` bằng `HybridSequential`.
Chúng ta bắt đầu với việc định nghĩa một mạng MLP đơn giản.


```{.python .input  n=3}
from d2l import mxnet as d2l
from mxnet import np, npx
from mxnet.gluon import nn
npx.set_np()

# factory for networks
def get_net():
    net = nn.HybridSequential()  
    net.add(nn.Dense(256, activation='relu'),
            nn.Dense(128, activation='relu'),
            nn.Dense(2))
    net.initialize()
    return net

x = np.random.normal(size=(1, 512))
net = get_net()
net(x)
```

<!--
By calling the `hybridize` function, we are able to compile and optimize the computation in the MLP. 
The model’s computation result remains unchanged.
-->

Bằng cách gọi hàm `hybridize`, ta có thể biên dịch và tối ưu hóa các tính toán trong MLP.
Kết quả tính toán của mô hình vẫn không thay đổi. 


```{.python .input  n=4}
net.hybridize()
net(x)
```

<!--
This seems almost too good to be true: simply designate a block to be `HybridSequential`, write the same code as before and invoke `hybridize`. 
Once this happens the network is optimized (we will benchmark the performance below). 
Unfortunately this does not work magically for every layer. 
That said, the blocks provided by Gluon are by default subclasses of `HybridBlock` and thus hybridizable. 
A layer will not be optimized if it inherits from the `Block` instead.
-->

Điều này có vẻ tốt đến mức khó tin: chỉ cần ta chỉ định một khối trở thành `HybridSequential`, sử dụng đoạn mã y hệt như trước và gọi hàm `hybridize`.
Một khi thực hiện xong những việc trên, mạng sẽ được tối ưu hóa (chúng ta sẽ đánh giá hiệu năng ở phía dưới).
Tiếc là cách này không hoạt động với mọi tầng.
Nhưng các khối được cung cấp sẵn bởi Gluon mặc định được kế thừa từ lớp `HybridBlock` và do đó có thể hybrid hóa được.
Tầng kế thừa từ lớp `Block` sẽ không thể tối ưu hóa được.


<!--
### Acceleration by Hybridization
-->

### Tăng tốc bằng Hybrid hóa

<!--
To demonstrate the performance improvement gained by compilation we compare the time needed to evaluate `net(x)` before and after hybridization. 
Let's define a function to measure this time first. 
It will come handy throughout the chapter as we set out to measure (and improve) performance.
-->

Để minh hoạ những cải thiện đạt được từ quá trình biên dịch, ta hãy so sánh thời gian cần thiết để đánh giá `net(x)` trước và sau phép hybrid hóa.
Đầu tiên hãy định nghĩa một hàm để đo thời gian trên.
Hàm này sẽ hữu ích trong suốt chương này khi chúng ta đo (và cải thiện) hiệu năng.


```{.python .input}
#@save
class Benchmark:    
    def __init__(self, description='Done'):
        self.description = description
        
    def __enter__(self):
        self.timer = d2l.Timer()
        return self

    def __exit__(self, *args):
        print(f'{self.description}: {self.timer.stop():.4f} sec')
```


<!--
Now we can invoke the network twice, once with and once without hybridization.
-->

Bây giờ ta có thể gọi mạng hai lần với có hybrid hóa và không hybrid hóa.


```{.python .input  n=5}
net = get_net()
with Benchmark('Without hybridization'):
    for i in range(1000): net(x)
    npx.waitall()

net.hybridize()
with Benchmark('With hybridization'):
    for i in range(1000): net(x)
    npx.waitall()
```


<!--
As is observed in the above results, after a HybridSequential instance calls the `hybridize` function, computing performance is improved through the use of symbolic programming.
-->

Như quan sát được trong các kết quả trên, sau khi thực thể HybridSequential gọi hàm `hybridize`, hiệu năng tính toán được cải thiện thông qua việc sử dụng lập trình ký hiệu.

<!--
### Serialization
-->

### Chuỗi hóa 
<!--https://itviec.com/blog/wp-content/uploads/download-manager-files/OOP_2013.pdf-->

<!--
One of the benefits of compiling the models is that we can serialize (save) the model and its parameters to disk. 
This allows us to store a model in a manner that is independent of the front-end language of choice. 
This allows us to deploy trained models to other devices and easily use other front-end programming languages. 
At the same time the code is often faster than what can be achieved in imperative programming. 
Let's see the `export` method in action.
-->

Một trong những lợi ích của việc biên dịch các mô hình là ta có thể chuỗi hóa (*serialize*) mô hình và các tham số mô hình để lưu trữ.
Điều này cho phép ta lưu trữ mô hình mà không phụ thuộc vào ngôn ngữ front-end.
Điều này cũng cho phép ta sử dụng các mô hình đã huấn luyện trên các thiết bị khác và dễ dàng sử dụng các ngôn ngữ lập trình front-end khác.
Đồng thời, mã nguồn này thường thực thi nhanh hơn so với khi lập trình mệnh lệnh.
Hãy xem xét phương thức `export` sau.


```{.python .input  n=13}
net.export('my_mlp')
!ls -lh my_mlp*
```


<!--
The model is decomposed into a (large binary) parameter file and a JSON description of the program required to execute to compute the model. 
The files can be read by other front-end languages supported by Python or MXNet, such as C++, R, Scala, and Perl. Let's have a look at the model description.
-->

Mô hình này được chia ra thành một tập tin (nhị phân) lớn chứa tham số và tập tin JSON mô tả cấu trúc mô hình.
Các tập tin có thể được đọc bởi các ngôn ngữ front-end khác được hỗ trợ bởi Python hoặc MXNet, ví dụ như C++, R, Scala, và Perl. Tập tin JSON có dạng như sau


```{.python .input  n=7}
!head my_mlp-symbol.json
```


<!--
Things are slightly more tricky when it comes to models that resemble code more closely. 
Basically hybridization needs to deal with control flow and Python overhead in a much more immediate manner. Moreover,
-->

Mọi thứ trở nên phức tạp hơn một chút khi làm việc với các mô hình gần với mã nguồn.
Về cơ bản, việc hybrid hóa cần giải quyết trực tiếp luồng điều khiển và các chi phí tính toán của Python.


<!--
Contrary to the Block instance, which needs to use the `forward` function, for a HybridBlock instance we need to use the `hybrid_forward` function.
-->

Hơn nữa, trong khi thực thể của lớp Block cần sử dụng hàm `forward`, thực thể của lớp HybridBlock lại sử dụng hàm `hybrid_forward`.



<!--
Earlier, we demonstrated that, after calling the `hybridize` method, the model is able to achieve superior computing performance and portability. 
Note, though that hybridization can affect model flexibility, in particular in terms of control flow. 
We will illustrate how to design more general models and also how compilation will remove spurious Python elements.
-->

Trên đây chúng ta thấy rằng phương thức `hybridize` có thể giúp mô hình đạt được hiệu năng tính toán và tính cơ động vượt trội hơn. 
Dù vậy, sự hybrid hóa có thể ảnh hưởng tới tính linh hoạt của mô hình, đặc biệt là trong điều khiển luồng. 
Ta sẽ minh họa cách thiết kế các mô hình tổng quát hơn cũng như cách trình biên dịch loại bỏ các thành phần thừa trong Python. 


```{.python .input  n=8}
class HybridNet(nn.HybridBlock):
    def __init__(self, **kwargs):
        super(HybridNet, self).__init__(**kwargs)
        self.hidden = nn.Dense(4)
        self.output = nn.Dense(2)

    def hybrid_forward(self, F, x):
        print('module F: ', F)
        print('value  x: ', x)
        x = F.npx.relu(self.hidden(x))
        print('result  : ', x)
        return self.output(x)
```


<!--
The code above implements a simple network with 4 hidden units and 2 outputs. 
`hybrid_forward` takes an additional argument - the module `F`. 
This is needed since, depending on whether the code has been hybridized or not, it will use a slightly different library (`ndarray` or `symbol`) for processing. 
Both classes perform very similar functions and MXNet automatically determines the argument. 
To understand what is going on we print the arguments as part of the function invocation.
-->

Đoạn mã trên biểu diễn một mạng đơn giản với 4 nút ẩn và 2 đầu ra. 
Phương thức `hybrid_foward` lấy thêm một đối số - mô-đun `F`.
Đối số này là cần thiết để chọn thư viện xử lý phù hợp (`ndarray` hoặc `symbol`) tùy vào việc chương trình có được hybrid hóa hay không.
Cả hai lớp này thực hiện các chức năng rất giống nhau và MXNet sẽ tự động xác định đối số đầu vào. 
Để hiểu chuyện gì đang diễn ra chúng ta sẽ in các đối số đầu vào khi gọi hàm. 


```{.python .input  n=9}
net = HybridNet()
net.initialize()
x = np.random.normal(size=(1, 3))
net(x)
```


<!--
Repeating the forward computation will lead to the same output (we omit details). 
Now let's see what happens if we invoke the `hybridize` method.
-->

Lặp lại nhiều lần việc tính lượt truyền xuôi sẽ cho ra cùng kết quả (ta bỏ qua chi tiết).
Bây giờ hãy xem chuyện gì xảy ra nếu ta kích hoạt phương thức `hybridize`. 


```{.python .input  n=10}
net.hybridize()
net(x)
```


<!--
Instead of using `ndarray` we now use the `symbol` module for `F`. 
Moreover, even though the input is of `ndarray` type, the data flowing through the network is now converted to `symbol` type as part of the compilation process. 
Repeating the function call leads to a surprising outcome:
-->

Thay vì `ndarray`, lúc này ta sử dụng mô-đun `symbol` cho `F`.
Thêm vào đó, mặc dù đầu vào thuộc kiểu `ndarray`, dữ liệu truyền qua mạng bây giờ được chuyển thành kiểu `symbol` như một phần của quá trình biên dịch.
Việc gọi lại hàm `net` dẫn tới một kết quả đáng kinh ngạc:


```{.python .input  n=11}
net(x)
```


<!--
This is quite different from what we saw previously. 
All print statements, as defined in `hybrid_forward` are omitted. 
Indeed, after hybridization the execution of `net(x)` does not involve the Python interpreter any longer. 
This means that any spurious Python code is omitted (such as print statements) in favor of a much more streamlined execution and better performance. 
Instead, MXNet directly calls the C++ backend. 
Also note that some functions are not supported in the `symbol` module (like `asnumpy`) and operations in-place like `a += b` and `a[:] = a + b` must be rewritten as `a = a + b`. 
Nonetheless, compilation of models is worth the effort whenever speed matters. 
The benefit can range from small percentage points to more than twice the speed, depending on the complexity of the model, the speed of the CPU and the speed and number of GPUs.
-->

Điều này khá khác biệt so vớinhững gì ta đã thấy trước đó.
Tất cả các lệnh in được định nghĩa trong `hybrid_forward` đều bị bỏ qua.
Thật vậy, sau khi hybrid hóa, việc thực thi lệnh `net(x)` không còn liên quan gì tới trình thông dịch của Python nữa.
Nghĩa là bất cứ đoạn mã Python nào không cần thiết cho tính toán sẽ bị bỏ qua (chẳng hạn như các lệnh in) để việc thực thi trôi chảy hơn và hiệu năng tốt hơn.
Và thay vì gọi Python, MXNet gọi trực tiếp back-end C++.  
Cũng nên lưu ý rằng một số hàm không được hỗ trợ trong mô-đun `symbol` (như `asnumpy`) và các toán tử thực thi tại chỗ (*in-place*) như `a += b` và `a[:] = a + b` phải được viết lại là `a = a + b`.
Tuy nhiên, việc biên dịch mô hình vẫn đáng để thực hiện bất cứ khi nào ta quan tâm đến tốc độ.
Lợi ích về tốc độ này có thể tăng từ vài phần trăm tới hơn hai lần, tùy thuộc vào sự phức tạp của mô hình, tốc độ của CPU, tốc độ và số lượng GPU.


<!--
## Summary
-->

## Tóm tắt

<!--
* Imperative programming makes it easy to design new models since it is possible to write code with control flow and the ability to use a large amount of the Python software ecosystem.
* Symbolic programming requires that we specify the program and compile it before executing it. The benefit is improved performance.
* MXNet is able to combine the advantages of both approaches as needed.
* Models constructed by the `HybridSequential` and `HybridBlock` classes are able to convert imperative programs into symbolic programs by calling the `hybridize` method.
-->

* Lập trình mệnh lệnh khiến việc thiết kế mô hình mới dễ dàng hơn vì ta có thể viết mã với luồng điều khiển và được sử dụng hệ sinh thái phần mềm của Python.
* Lập trình ký hiệu đòi hỏi chúng ta định nghĩa và biên dịch chương trình trước khi thực thi nó. Lợi ích là hiệu năng được cải thiện.
* MXNet có thể kết hợp những ưu điểm của cả hai phương pháp khi cần thiết.
* Mô hình được xây dựng bởi các lớp `HybridSequential` và `HybridBlock` có thể chuyển đổi các chương trình mệnh lệnh thành các chương trình ký hiệu bằng cách gọi phương thức `hybridize`.


<!--
## Exercises
-->

## Bài tập

<!--
1. Design a network using the `HybridConcurrent` class. Alternatively look at :ref:`sec_googlenet` for a network to compose.
2. Add `x.asnumpy()` to the first line of the `hybrid_forward` function of the HybridNet class in this section. Execute the code and observe the errors you encounter. Why do they happen?
3. What happens if we add control flow, i.e., the Python statements `if` and `for` in the `hybrid_forward` function?
4. Review the models that interest you in the previous chapters and use the HybridBlock class or HybridSequential class to implement them.
-->

1. Hãy thiết kế một mạng bằng cách sử dụng lớp `HybridConcurrent`, có thể thử với GoogleNet trong :ref: `sec_googlenet`.
2. Hãy thêm `x.asnumpy()` vào dòng đầu tiên của hàm `hybrid_forward` trong lớp HybridNet, rồi thực thi mã nguồn và quan sát các lỗi bạn gặp phải. Tại sao các lỗi này xảy ra?
3. Điều gì sẽ xảy ra nếu ta thêm luồng điều khiển, cụ thể là các lệnh Python `if` và `for` trong hàm `hybrid_forward`?
4. Hãy lập trình các mô hình bạn thích trong các chương trước bằng cách sử dụng lớp HybridBlock hoặc HybridSequential.


## Thảo luận
* [Tiếng Anh - MXNet](https://discuss.d2l.ai/t/360)
* [Tiếng Việt](https://forum.machinelearningcoban.com/c/d2l)

### Những người thực hiện
Bản dịch trong trang này được thực hiện bởi:

* Đoàn Võ Duy Thanh
* Nguyễn Văn Tâm
* Phạm Hồng Vinh
* Lê Khắc Hồng Phúc
* Nguyễn Văn Quang
* Nguyễn Mai Hoàng Long
* Phạm Minh Đức
* Nguyễn Văn Cường
