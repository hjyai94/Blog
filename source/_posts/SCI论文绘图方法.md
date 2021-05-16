---
title: SCI论文绘图方法
tags: gnuplot
categories: 工作
abbrlink: 49149
date: 2019-06-13 14:24:53
---
小同学，也许你在看论文时也困惑过别人的图为何画的如此的美，如此的精细，不用担心，这里教你SCI级别的论文绘图方法，一看就会，动手就忘（划掉，当然，你可以联系我帮你画，价格好商量）。

# 软件介绍
我们这里使用的软件叫做gnuplot，这是命令行驱动的绘图工具，可以讲数学函数或数值资料以平面图或者立体图的形式在不同种类的终端机或者是绘图输出装置上。另外，gnuplot是开源的，不必担心版权问题。

## 软件安装
Windows平台下使用gnuplot非常简单，只要到[gnuplot官网](http://www.gnuplot.info/)上找到windows版本下载安装即可，双击打开安装后的软件，是如下图的界面。
![gnu_terminal](https://raw.githubusercontent.com/hjyai94/Blog/master/source/uploads/gnuplot/gnu_terminal.png)

# 开始绘图
我们只需要在`>gnuplot`后面写命令就可以画图了，如果我们想要画一个正弦曲线，我们只需要输入下面的命令即可；
```
plot sin(5*x)
```
这样我们就得到了一个绘制的正弦曲线，当然我们的目标是为了绘制出SCI论文级的图，当然不能满足于此了。
下面我们来真正开始看如何绘制一个SCI级别论文图。

# SCI绘图
下面我们来看一个实际的绘图的情况，下面我用实际训练神经网络的损失函数来举例使用gnuplot绘制的方法。
首先我们有数据文件，这个数据文件我们命名为 $loss.dat$，保存在`E:\plot`下，数据的格式如下图所示：
![data](https://raw.githubusercontent.com/hjyai94/Blog/master/source/uploads/gnuplot/data.png)
下面我们使用`plot`命令来画我们这个文件的数据，首先我们需要切换到数据文件的保存路径，也就是`E:\plot`，大家可以使用软件的`CHDir`切换，也可以使用`cd "E:\plot"`切换到数据保存文件的路径。
使用如下的命令就能绘制出这个文件的图了：
```
plot "loss.dat" using 1:2 with lines title "CNN"
```
这行代码表示，我们使用`loss.dat`的第1、2列数据绘制线，并且设置图例为`title`。
我们会绘制出下图的结果：
![plot1](https://raw.githubusercontent.com/hjyai94/Blog/master/source/uploads/gnuplot/plot1.png)
上面的图是个不完整的图，它没有横纵坐标的描述，也没有图的标题，下面我们来设置这些：
```
set xlabel "Epoch" 
set ylabel "Cross Entropy" 
set title "Traing Loss"
replot
```
将上面的命令以此执行，就能将横坐标设置为"Epoch"，纵坐标设置为"Cross Entropy"，设置图的标题为"Traing Loss"。`replot`将图重新画一遍，这样我们就会把我们增加内容绘制在图当中了。
![plot2](https://raw.githubusercontent.com/hjyai94/Blog/master/source/uploads/gnuplot/plot2.png)
上面的图我们已经基本上画好，图的各种元素已经完备了，但是仍然不好看，所以下面我们来进一步优化我们的图，选一篇我前面在看的文章，Dolz等人设计的HyperDense网络[2]，我们按照他们的绘图风格来进一步优化我们的图。
对比他们的损失函数的图(文献[2]中的图3，如下图所示)，我们可以看出，我们的图名太小，纵坐标的名字不够清晰，图例没有加框，曲线不够平滑。另外我们绘制的是损失函数的收敛曲线，我们应该让其看起来收敛(虽然实际上选取数据的这段还没有收敛)。
![hyperdense](https://raw.githubusercontent.com/hjyai94/Blog/master/source/uploads/gnuplot/hyperdense.png)
首先我们添加网格
```
set grid
```
然后，将横纵坐标刻度设置为"Times New Roman"，10号字，横纵坐标名设置为"Times New Roman"，14号字。设置图例为12号字。
```
set xtics font 'times.ttf,10'
set ytics font 'times.ttf,10'
set xlabel 'Epoch' font 'times.ttf,14'
set ylabel 'Cross Entropy' font 'times.ttf,14'
set title 'Traing Loss' font 'times.ttf,14'
set key font 'times.ttf,12'
```
![plot3](https://raw.githubusercontent.com/hjyai94/Blog/master/source/uploads/gnuplot/plot3.png)
这时我们再看我们画的图，已经基本具有一般学术论文的样子了。但是，我们仍有可以提高的地方，比如说，我们的线条颜色不好看，我们的收敛曲线看不出收敛的趋势。
更改线条的颜色，我们改成红色：
```
set style line 1 lw 1.5 lc rgb "#F62217"
plot file using 1:2 with lines ls 1 title "CNN"
```
另外常用的两种线条颜色，一个是蓝色一个是黄色。
```
set style line 2 linewidth 2 linecolor rgb "#D4A017"
set style line 3 linewidth 3 linecolor rgb "#2B60DE"
```
将纵坐标设范围为0.04-0.10，这样我们曲线更加集中，可以更好地看出收敛趋势。
```
set yrange [0.04:0.10]
```
下面我们为图例加框，设置图例位置，图例框宽度为2，高度为1，文字为左对齐，线段在左文字在右，并且设置图例线段长度为2。
```
set key box
set key center at 42,0.09
set key width 2
set key height 1
set key Left
set key reverse
set key samplen 2
```
经过上面一顿猛如虎的操作，我们已经基本上绘制出了可以放在论文里面的图了，下面是我们绘制的结果：
![plot4](https://raw.githubusercontent.com/hjyai94/Blog/master/source/uploads/gnuplot/plot4.png)
但是，我们仍能优化这个图，我们希望它能更加的平滑。下面我们使它更加平滑，在gnuplot中的数据平滑命令为smooth，然后常用的算法为$bezier$和esplines$，因为$splines$要额外的平滑权重，这里我们就使用不需要平滑权重的bezier算法。
```
plot ”loss.dat“ using 1:2 with lines ls 1 smooth bezier title "CNN"
```
这就可以得到我们最终需要的曲线图了。
![result](https://raw.githubusercontent.com/hjyai94/Blog/master/source/uploads/gnuplot/result.png)

# 利用脚本绘图
聪明的你一定在想，难道我每次都需要一行一行的输入命令吗？其实我们完全可以使用脚本，将前面的命令写进脚本中，然后一次运行就可以得到我们最终的绘图结果了。首先我们需要建立一个文件，文件名为$loss.gnu$，将这个放置在于数据相同的文件夹内，我们这里是`E:\plot`。
脚本的内容如下：
```
set encoding utf8
set term wxt enhanced

set xtics font 'times.ttf,10'
set ytics font 'times.ttf,10'
set xlabel 'Epoch' font 'times.ttf,14'
set ylabel 'Cross Entropy' font 'times.ttf,14'
set title 'Traing Loss' font 'times.ttf,14'
set key font 'times.ttf,12'
set key center at 42,0.09
set key box
set key reverse
set key width 2
set key height 1
set key Left
set key samplen 2
set grid
set yrange [0.04:0.10]
set style line 1 lw 1.5 lc rgb "#F62217"

file = "loss.dat"
plot file using 1:2 with lines ls 1 smooth bezier title "CNN"
```
我们在命令行里面输入`load loss.gnu`运行整个脚本，得到我们最后想要的图片了。
至此，我们关于使用gnuplot绘制SCI论文图的方法就介绍完毕了，如果大家有需要可以参考文献[1]的内容了解更多。


# 参考文献
[1] http://vision.ouc.edu.cn/zhenghaiyong/courses/tutorials/gnuplot/gnuplot-zh.pdf
[2] Dolz J, Gopinath K, Yuan J, et al. HyperDense-Net: A hyper-densely connected CNN for multi-modal image segmentation[J]. IEEE transactions on medical imaging, 2018.




