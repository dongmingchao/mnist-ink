åž
Ø))
:
Add
x"T
y"T
z"T"
Ttype:
2	
ī
	ApplyAdam
var"T	
m"T	
v"T
beta1_power"T
beta2_power"T
lr"T

beta1"T

beta2"T
epsilon"T	
grad"T
out"T" 
Ttype:
2	"
use_lockingbool( "
use_nesterovbool( 

ArgMax

input"T
	dimension"Tidx
output"output_type" 
Ttype:
2	"
Tidxtype0:
2	"
output_typetype0	:
2	
x
Assign
ref"T

value"T

output_ref"T"	
Ttype"
validate_shapebool("
use_lockingbool(
R
BroadcastGradientArgs
s0"T
s1"T
r0"T
r1"T"
Ttype0:
2	
8
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype
8
Const
output"dtype"
valuetensor"
dtypetype
ė
Conv2D

input"T
filter"T
output"T"
Ttype:
2"
strides	list(int)"
use_cudnn_on_gpubool(""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)


Conv2DBackpropFilter

input"T
filter_sizes
out_backprop"T
output"T"
Ttype:
2"
strides	list(int)"
use_cudnn_on_gpubool(""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)


Conv2DBackpropInput
input_sizes
filter"T
out_backprop"T
output"T"
Ttype:
2"
strides	list(int)"
use_cudnn_on_gpubool(""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

S
DynamicStitch
indices*N
data"T*N
merged"T"
Nint(0"	
Ttype
4
Fill
dims

value"T
output"T"	
Ttype
,
Floor
x"T
y"T"
Ttype:
2
?
FloorDiv
x"T
y"T
z"T"
Ttype:
2	
8
FloorMod
x"T
y"T
z"T"
Ttype:	
2	
V
HistogramSummary
tag
values"T
summary"
Ttype0:
2	
.
Identity

input"T
output"T"	
Ttype
,
Log
x"T
y"T"
Ttype:

2
p
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:
	2
Ō
MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0""
paddingstring:
SAMEVALID":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
ī
MaxPoolGrad

orig_input"T
orig_output"T	
grad"T
output"T"
ksize	list(int)(0"
strides	list(int)(0""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW"
Ttype0:
2	
;
Maximum
x"T
y"T
z"T"
Ttype:

2	

Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
8
MergeSummary
inputs*N
summary"
Nint(0
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(
=
Mul
x"T
y"T
z"T"
Ttype:
2	
.
Neg
x"T
y"T"
Ttype:

2	

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:

Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
~
RandomUniform

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	
a
Range
start"Tidx
limit"Tidx
delta"Tidx
output"Tidx"
Tidxtype0:	
2	
>
RealDiv
x"T
y"T
z"T"
Ttype:
2	
5

Reciprocal
x"T
y"T"
Ttype:

2	
D
Relu
features"T
activations"T"
Ttype:
2	
V
ReluGrad
	gradients"T
features"T
	backprops"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
P
ScalarSummary
tags
values"T
summary"
Ttype:
2	
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
e
ShapeN
input"T*N
output"out_type*N"
Nint(0"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
:
Sub
x"T
y"T
z"T"
Ttype:
2	

Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
c
Tile

input"T
	multiples"
Tmultiples
output"T"	
Ttype"

Tmultiplestype0:
2	

TruncatedNormal

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	
s

VariableV2
ref"dtype"
shapeshape"
dtypetype"
	containerstring "
shared_namestring "serve*1.5.02
b'unknown'æ„
s
inputs/x_inputPlaceholder*
shape:’’’’’’’’’*
dtype0*(
_output_shapes
:’’’’’’’’’
q
inputs/y_inputPlaceholder*
shape:’’’’’’’’’
*
dtype0*'
_output_shapes
:’’’’’’’’’

S
inputs/dropoutPlaceholder*
shape:*
dtype0*
_output_shapes
:
j
image_input/shapeConst*%
valueB"’’’’         *
dtype0*
_output_shapes
:

image_inputReshapeinputs/x_inputimage_input/shape*
T0*
Tshape0*/
_output_shapes
:’’’’’’’’’

-hidden_layer_1/weights/truncated_normal/shapeConst*%
valueB"             *
dtype0*
_output_shapes
:
q
,hidden_layer_1/weights/truncated_normal/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
s
.hidden_layer_1/weights/truncated_normal/stddevConst*
valueB
 *ĶĢĢ=*
dtype0*
_output_shapes
: 
Š
7hidden_layer_1/weights/truncated_normal/TruncatedNormalTruncatedNormal-hidden_layer_1/weights/truncated_normal/shape*

seed *
T0*
dtype0*&
_output_shapes
: *
seed2 
Ģ
+hidden_layer_1/weights/truncated_normal/mulMul7hidden_layer_1/weights/truncated_normal/TruncatedNormal.hidden_layer_1/weights/truncated_normal/stddev*
T0*&
_output_shapes
: 
ŗ
'hidden_layer_1/weights/truncated_normalAdd+hidden_layer_1/weights/truncated_normal/mul,hidden_layer_1/weights/truncated_normal/mean*
T0*&
_output_shapes
: 
£
hidden_layer_1/weights/Variable
VariableV2*
shape: *
shared_name *
dtype0*&
_output_shapes
: *
	container 

&hidden_layer_1/weights/Variable/AssignAssignhidden_layer_1/weights/Variable'hidden_layer_1/weights/truncated_normal*
use_locking(*
T0*2
_class(
&$loc:@hidden_layer_1/weights/Variable*
validate_shape(*&
_output_shapes
: 
¶
$hidden_layer_1/weights/Variable/readIdentityhidden_layer_1/weights/Variable*
T0*2
_class(
&$loc:@hidden_layer_1/weights/Variable*&
_output_shapes
: 

"hidden_layer_1/weights/weights/tagConst*/
value&B$ Bhidden_layer_1/weights/weights*
dtype0*
_output_shapes
: 

hidden_layer_1/weights/weightsHistogramSummary"hidden_layer_1/weights/weights/tag$hidden_layer_1/weights/Variable/read*
T0*
_output_shapes
: 
h
hidden_layer_1/biases/ConstConst*
valueB *ĶĢĢ=*
dtype0*
_output_shapes
: 

hidden_layer_1/biases/Variable
VariableV2*
shape: *
shared_name *
dtype0*
_output_shapes
: *
	container 
ķ
%hidden_layer_1/biases/Variable/AssignAssignhidden_layer_1/biases/Variablehidden_layer_1/biases/Const*
use_locking(*
T0*1
_class'
%#loc:@hidden_layer_1/biases/Variable*
validate_shape(*
_output_shapes
: 
§
#hidden_layer_1/biases/Variable/readIdentityhidden_layer_1/biases/Variable*
T0*1
_class'
%#loc:@hidden_layer_1/biases/Variable*
_output_shapes
: 
}
 hidden_layer_1/biases/biases/tagConst*-
value$B" Bhidden_layer_1/biases/biases*
dtype0*
_output_shapes
: 

hidden_layer_1/biases/biasesHistogramSummary hidden_layer_1/biases/biases/tag#hidden_layer_1/biases/Variable/read*
T0*
_output_shapes
: 
ś
hidden_layer_1/Conv2DConv2Dimage_input$hidden_layer_1/weights/Variable/read*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*/
_output_shapes
:’’’’’’’’’ 

hidden_layer_1/addAddhidden_layer_1/Conv2D#hidden_layer_1/biases/Variable/read*
T0*/
_output_shapes
:’’’’’’’’’ 
i
hidden_layer_1/ReluReluhidden_layer_1/add*
T0*/
_output_shapes
:’’’’’’’’’ 
Ā
hidden_layer_1/MaxPoolMaxPoolhidden_layer_1/Relu*
T0*
strides
*
data_formatNHWC*
ksize
*
paddingSAME*/
_output_shapes
:’’’’’’’’’ 

-hidden_layer_2/weights/truncated_normal/shapeConst*%
valueB"          @   *
dtype0*
_output_shapes
:
q
,hidden_layer_2/weights/truncated_normal/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
s
.hidden_layer_2/weights/truncated_normal/stddevConst*
valueB
 *ĶĢĢ=*
dtype0*
_output_shapes
: 
Š
7hidden_layer_2/weights/truncated_normal/TruncatedNormalTruncatedNormal-hidden_layer_2/weights/truncated_normal/shape*

seed *
T0*
dtype0*&
_output_shapes
: @*
seed2 
Ģ
+hidden_layer_2/weights/truncated_normal/mulMul7hidden_layer_2/weights/truncated_normal/TruncatedNormal.hidden_layer_2/weights/truncated_normal/stddev*
T0*&
_output_shapes
: @
ŗ
'hidden_layer_2/weights/truncated_normalAdd+hidden_layer_2/weights/truncated_normal/mul,hidden_layer_2/weights/truncated_normal/mean*
T0*&
_output_shapes
: @
£
hidden_layer_2/weights/Variable
VariableV2*
shape: @*
shared_name *
dtype0*&
_output_shapes
: @*
	container 

&hidden_layer_2/weights/Variable/AssignAssignhidden_layer_2/weights/Variable'hidden_layer_2/weights/truncated_normal*
use_locking(*
T0*2
_class(
&$loc:@hidden_layer_2/weights/Variable*
validate_shape(*&
_output_shapes
: @
¶
$hidden_layer_2/weights/Variable/readIdentityhidden_layer_2/weights/Variable*
T0*2
_class(
&$loc:@hidden_layer_2/weights/Variable*&
_output_shapes
: @

"hidden_layer_2/weights/weights/tagConst*/
value&B$ Bhidden_layer_2/weights/weights*
dtype0*
_output_shapes
: 

hidden_layer_2/weights/weightsHistogramSummary"hidden_layer_2/weights/weights/tag$hidden_layer_2/weights/Variable/read*
T0*
_output_shapes
: 
h
hidden_layer_2/biases/ConstConst*
valueB@*ĶĢĢ=*
dtype0*
_output_shapes
:@

hidden_layer_2/biases/Variable
VariableV2*
shape:@*
shared_name *
dtype0*
_output_shapes
:@*
	container 
ķ
%hidden_layer_2/biases/Variable/AssignAssignhidden_layer_2/biases/Variablehidden_layer_2/biases/Const*
use_locking(*
T0*1
_class'
%#loc:@hidden_layer_2/biases/Variable*
validate_shape(*
_output_shapes
:@
§
#hidden_layer_2/biases/Variable/readIdentityhidden_layer_2/biases/Variable*
T0*1
_class'
%#loc:@hidden_layer_2/biases/Variable*
_output_shapes
:@
}
 hidden_layer_2/biases/biases/tagConst*-
value$B" Bhidden_layer_2/biases/biases*
dtype0*
_output_shapes
: 

hidden_layer_2/biases/biasesHistogramSummary hidden_layer_2/biases/biases/tag#hidden_layer_2/biases/Variable/read*
T0*
_output_shapes
: 

hidden_layer_2/Conv2DConv2Dhidden_layer_1/MaxPool$hidden_layer_2/weights/Variable/read*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*/
_output_shapes
:’’’’’’’’’@

hidden_layer_2/addAddhidden_layer_2/Conv2D#hidden_layer_2/biases/Variable/read*
T0*/
_output_shapes
:’’’’’’’’’@
i
hidden_layer_2/ReluReluhidden_layer_2/add*
T0*/
_output_shapes
:’’’’’’’’’@
Ā
hidden_layer_2/MaxPoolMaxPoolhidden_layer_2/Relu*
T0*
strides
*
data_formatNHWC*
ksize
*
paddingSAME*/
_output_shapes
:’’’’’’’’’@

0full_connection_1/weights/truncated_normal/shapeConst*
valueB"@     *
dtype0*
_output_shapes
:
t
/full_connection_1/weights/truncated_normal/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
v
1full_connection_1/weights/truncated_normal/stddevConst*
valueB
 *ĶĢĢ=*
dtype0*
_output_shapes
: 
Š
:full_connection_1/weights/truncated_normal/TruncatedNormalTruncatedNormal0full_connection_1/weights/truncated_normal/shape*

seed *
T0*
dtype0* 
_output_shapes
:
Ą*
seed2 
Ļ
.full_connection_1/weights/truncated_normal/mulMul:full_connection_1/weights/truncated_normal/TruncatedNormal1full_connection_1/weights/truncated_normal/stddev*
T0* 
_output_shapes
:
Ą
½
*full_connection_1/weights/truncated_normalAdd.full_connection_1/weights/truncated_normal/mul/full_connection_1/weights/truncated_normal/mean*
T0* 
_output_shapes
:
Ą

"full_connection_1/weights/Variable
VariableV2*
shape:
Ą*
shared_name *
dtype0* 
_output_shapes
:
Ą*
	container 

)full_connection_1/weights/Variable/AssignAssign"full_connection_1/weights/Variable*full_connection_1/weights/truncated_normal*
use_locking(*
T0*5
_class+
)'loc:@full_connection_1/weights/Variable*
validate_shape(* 
_output_shapes
:
Ą
¹
'full_connection_1/weights/Variable/readIdentity"full_connection_1/weights/Variable*
T0*5
_class+
)'loc:@full_connection_1/weights/Variable* 
_output_shapes
:
Ą

%full_connection_1/weights/weights/tagConst*2
value)B' B!full_connection_1/weights/weights*
dtype0*
_output_shapes
: 
¦
!full_connection_1/weights/weightsHistogramSummary%full_connection_1/weights/weights/tag'full_connection_1/weights/Variable/read*
T0*
_output_shapes
: 
m
full_connection_1/biases/ConstConst*
valueB*ĶĢĢ=*
dtype0*
_output_shapes	
:

!full_connection_1/biases/Variable
VariableV2*
shape:*
shared_name *
dtype0*
_output_shapes	
:*
	container 
ś
(full_connection_1/biases/Variable/AssignAssign!full_connection_1/biases/Variablefull_connection_1/biases/Const*
use_locking(*
T0*4
_class*
(&loc:@full_connection_1/biases/Variable*
validate_shape(*
_output_shapes	
:
±
&full_connection_1/biases/Variable/readIdentity!full_connection_1/biases/Variable*
T0*4
_class*
(&loc:@full_connection_1/biases/Variable*
_output_shapes	
:

#full_connection_1/biases/biases/tagConst*0
value'B% Bfull_connection_1/biases/biases*
dtype0*
_output_shapes
: 
”
full_connection_1/biases/biasesHistogramSummary#full_connection_1/biases/biases/tag&full_connection_1/biases/Variable/read*
T0*
_output_shapes
: 
p
full_connection_1/Reshape/shapeConst*
valueB"’’’’@  *
dtype0*
_output_shapes
:

full_connection_1/ReshapeReshapehidden_layer_2/MaxPoolfull_connection_1/Reshape/shape*
T0*
Tshape0*(
_output_shapes
:’’’’’’’’’Ą
æ
full_connection_1/MatMulMatMulfull_connection_1/Reshape'full_connection_1/weights/Variable/read*
transpose_b( *
T0*(
_output_shapes
:’’’’’’’’’*
transpose_a( 
k
full_connection_1/ReluRelufull_connection_1/MatMul*
T0*(
_output_shapes
:’’’’’’’’’
h
#full_connection_1/dropout/keep_probConst*
valueB
 *   ?*
dtype0*
_output_shapes
: 
u
full_connection_1/dropout/ShapeShapefull_connection_1/Relu*
T0*
out_type0*
_output_shapes
:
q
,full_connection_1/dropout/random_uniform/minConst*
valueB
 *    *
dtype0*
_output_shapes
: 
q
,full_connection_1/dropout/random_uniform/maxConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
Į
6full_connection_1/dropout/random_uniform/RandomUniformRandomUniformfull_connection_1/dropout/Shape*

seed *
T0*
dtype0*(
_output_shapes
:’’’’’’’’’*
seed2 
°
,full_connection_1/dropout/random_uniform/subSub,full_connection_1/dropout/random_uniform/max,full_connection_1/dropout/random_uniform/min*
T0*
_output_shapes
: 
Ģ
,full_connection_1/dropout/random_uniform/mulMul6full_connection_1/dropout/random_uniform/RandomUniform,full_connection_1/dropout/random_uniform/sub*
T0*(
_output_shapes
:’’’’’’’’’
¾
(full_connection_1/dropout/random_uniformAdd,full_connection_1/dropout/random_uniform/mul,full_connection_1/dropout/random_uniform/min*
T0*(
_output_shapes
:’’’’’’’’’
¦
full_connection_1/dropout/addAdd#full_connection_1/dropout/keep_prob(full_connection_1/dropout/random_uniform*
T0*(
_output_shapes
:’’’’’’’’’
z
full_connection_1/dropout/FloorFloorfull_connection_1/dropout/add*
T0*(
_output_shapes
:’’’’’’’’’

full_connection_1/dropout/divRealDivfull_connection_1/Relu#full_connection_1/dropout/keep_prob*
T0*(
_output_shapes
:’’’’’’’’’

full_connection_1/dropout/mulMulfull_connection_1/dropout/divfull_connection_1/dropout/Floor*
T0*(
_output_shapes
:’’’’’’’’’

0full_connection_2/weights/truncated_normal/shapeConst*
valueB"   
   *
dtype0*
_output_shapes
:
t
/full_connection_2/weights/truncated_normal/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
v
1full_connection_2/weights/truncated_normal/stddevConst*
valueB
 *ĶĢĢ=*
dtype0*
_output_shapes
: 
Ļ
:full_connection_2/weights/truncated_normal/TruncatedNormalTruncatedNormal0full_connection_2/weights/truncated_normal/shape*

seed *
T0*
dtype0*
_output_shapes
:	
*
seed2 
Ī
.full_connection_2/weights/truncated_normal/mulMul:full_connection_2/weights/truncated_normal/TruncatedNormal1full_connection_2/weights/truncated_normal/stddev*
T0*
_output_shapes
:	

¼
*full_connection_2/weights/truncated_normalAdd.full_connection_2/weights/truncated_normal/mul/full_connection_2/weights/truncated_normal/mean*
T0*
_output_shapes
:	


"full_connection_2/weights/Variable
VariableV2*
shape:	
*
shared_name *
dtype0*
_output_shapes
:	
*
	container 

)full_connection_2/weights/Variable/AssignAssign"full_connection_2/weights/Variable*full_connection_2/weights/truncated_normal*
use_locking(*
T0*5
_class+
)'loc:@full_connection_2/weights/Variable*
validate_shape(*
_output_shapes
:	

ø
'full_connection_2/weights/Variable/readIdentity"full_connection_2/weights/Variable*
T0*5
_class+
)'loc:@full_connection_2/weights/Variable*
_output_shapes
:	


%full_connection_2/weights/weights/tagConst*2
value)B' B!full_connection_2/weights/weights*
dtype0*
_output_shapes
: 
¦
!full_connection_2/weights/weightsHistogramSummary%full_connection_2/weights/weights/tag'full_connection_2/weights/Variable/read*
T0*
_output_shapes
: 
k
full_connection_2/biases/ConstConst*
valueB
*ĶĢĢ=*
dtype0*
_output_shapes
:


!full_connection_2/biases/Variable
VariableV2*
shape:
*
shared_name *
dtype0*
_output_shapes
:
*
	container 
ł
(full_connection_2/biases/Variable/AssignAssign!full_connection_2/biases/Variablefull_connection_2/biases/Const*
use_locking(*
T0*4
_class*
(&loc:@full_connection_2/biases/Variable*
validate_shape(*
_output_shapes
:

°
&full_connection_2/biases/Variable/readIdentity!full_connection_2/biases/Variable*
T0*4
_class*
(&loc:@full_connection_2/biases/Variable*
_output_shapes
:


#full_connection_2/biases/biases/tagConst*0
value'B% Bfull_connection_2/biases/biases*
dtype0*
_output_shapes
: 
”
full_connection_2/biases/biasesHistogramSummary#full_connection_2/biases/biases/tag&full_connection_2/biases/Variable/read*
T0*
_output_shapes
: 
°
MatMulMatMulfull_connection_1/dropout/mul'full_connection_2/weights/Variable/read*
transpose_b( *
T0*'
_output_shapes
:’’’’’’’’’
*
transpose_a( 
l
addAddMatMul&full_connection_2/biases/Variable/read*
T0*'
_output_shapes
:’’’’’’’’’

I
SoftmaxSoftmaxadd*
T0*'
_output_shapes
:’’’’’’’’’

V
predict_op/dimensionConst*
value	B :*
dtype0*
_output_shapes
: 


predict_opArgMaxSoftmaxpredict_op/dimension*

Tidx0*
T0*
output_type0	*#
_output_shapes
:’’’’’’’’’
E
LogLogSoftmax*
T0*'
_output_shapes
:’’’’’’’’’

Q
mulMulinputs/y_inputLog*
T0*'
_output_shapes
:’’’’’’’’’

_
Sum/reduction_indicesConst*
valueB:*
dtype0*
_output_shapes
:
q
SumSummulSum/reduction_indices*

Tidx0*
	keep_dims( *
T0*#
_output_shapes
:’’’’’’’’’
=
NegNegSum*
T0*#
_output_shapes
:’’’’’’’’’
O
ConstConst*
valueB: *
dtype0*
_output_shapes
:
V
MeanMeanNegConst*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
N
	loss/tagsConst*
valueB
 Bloss*
dtype0*
_output_shapes
: 
G
lossScalarSummary	loss/tagsMean*
T0*
_output_shapes
: 
X
train/gradients/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
^
train/gradients/grad_ys_0Const*
valueB
 *  ?*
dtype0*
_output_shapes
: 
o
train/gradients/FillFilltrain/gradients/Shapetrain/gradients/grad_ys_0*
T0*
_output_shapes
: 
q
'train/gradients/Mean_grad/Reshape/shapeConst*
valueB:*
dtype0*
_output_shapes
:

!train/gradients/Mean_grad/ReshapeReshapetrain/gradients/Fill'train/gradients/Mean_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes
:
b
train/gradients/Mean_grad/ShapeShapeNeg*
T0*
out_type0*
_output_shapes
:
Ŗ
train/gradients/Mean_grad/TileTile!train/gradients/Mean_grad/Reshapetrain/gradients/Mean_grad/Shape*

Tmultiples0*
T0*#
_output_shapes
:’’’’’’’’’
d
!train/gradients/Mean_grad/Shape_1ShapeNeg*
T0*
out_type0*
_output_shapes
:
d
!train/gradients/Mean_grad/Shape_2Const*
valueB *
dtype0*
_output_shapes
: 
i
train/gradients/Mean_grad/ConstConst*
valueB: *
dtype0*
_output_shapes
:
Ø
train/gradients/Mean_grad/ProdProd!train/gradients/Mean_grad/Shape_1train/gradients/Mean_grad/Const*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
k
!train/gradients/Mean_grad/Const_1Const*
valueB: *
dtype0*
_output_shapes
:
¬
 train/gradients/Mean_grad/Prod_1Prod!train/gradients/Mean_grad/Shape_2!train/gradients/Mean_grad/Const_1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
e
#train/gradients/Mean_grad/Maximum/yConst*
value	B :*
dtype0*
_output_shapes
: 

!train/gradients/Mean_grad/MaximumMaximum train/gradients/Mean_grad/Prod_1#train/gradients/Mean_grad/Maximum/y*
T0*
_output_shapes
: 

"train/gradients/Mean_grad/floordivFloorDivtrain/gradients/Mean_grad/Prod!train/gradients/Mean_grad/Maximum*
T0*
_output_shapes
: 
z
train/gradients/Mean_grad/CastCast"train/gradients/Mean_grad/floordiv*

SrcT0*
_output_shapes
: *

DstT0

!train/gradients/Mean_grad/truedivRealDivtrain/gradients/Mean_grad/Tiletrain/gradients/Mean_grad/Cast*
T0*#
_output_shapes
:’’’’’’’’’
t
train/gradients/Neg_grad/NegNeg!train/gradients/Mean_grad/truediv*
T0*#
_output_shapes
:’’’’’’’’’
a
train/gradients/Sum_grad/ShapeShapemul*
T0*
out_type0*
_output_shapes
:

train/gradients/Sum_grad/SizeConst*
value	B :*1
_class'
%#loc:@train/gradients/Sum_grad/Shape*
dtype0*
_output_shapes
: 
±
train/gradients/Sum_grad/addAddSum/reduction_indicestrain/gradients/Sum_grad/Size*
T0*1
_class'
%#loc:@train/gradients/Sum_grad/Shape*
_output_shapes
:
½
train/gradients/Sum_grad/modFloorModtrain/gradients/Sum_grad/addtrain/gradients/Sum_grad/Size*
T0*1
_class'
%#loc:@train/gradients/Sum_grad/Shape*
_output_shapes
:

 train/gradients/Sum_grad/Shape_1Const*
valueB:*1
_class'
%#loc:@train/gradients/Sum_grad/Shape*
dtype0*
_output_shapes
:

$train/gradients/Sum_grad/range/startConst*
value	B : *1
_class'
%#loc:@train/gradients/Sum_grad/Shape*
dtype0*
_output_shapes
: 

$train/gradients/Sum_grad/range/deltaConst*
value	B :*1
_class'
%#loc:@train/gradients/Sum_grad/Shape*
dtype0*
_output_shapes
: 
ķ
train/gradients/Sum_grad/rangeRange$train/gradients/Sum_grad/range/starttrain/gradients/Sum_grad/Size$train/gradients/Sum_grad/range/delta*

Tidx0*1
_class'
%#loc:@train/gradients/Sum_grad/Shape*
_output_shapes
:

#train/gradients/Sum_grad/Fill/valueConst*
value	B :*1
_class'
%#loc:@train/gradients/Sum_grad/Shape*
dtype0*
_output_shapes
: 
Ä
train/gradients/Sum_grad/FillFill train/gradients/Sum_grad/Shape_1#train/gradients/Sum_grad/Fill/value*
T0*1
_class'
%#loc:@train/gradients/Sum_grad/Shape*
_output_shapes
:

&train/gradients/Sum_grad/DynamicStitchDynamicStitchtrain/gradients/Sum_grad/rangetrain/gradients/Sum_grad/modtrain/gradients/Sum_grad/Shapetrain/gradients/Sum_grad/Fill*
T0*1
_class'
%#loc:@train/gradients/Sum_grad/Shape*
N*#
_output_shapes
:’’’’’’’’’

"train/gradients/Sum_grad/Maximum/yConst*
value	B :*1
_class'
%#loc:@train/gradients/Sum_grad/Shape*
dtype0*
_output_shapes
: 
Ų
 train/gradients/Sum_grad/MaximumMaximum&train/gradients/Sum_grad/DynamicStitch"train/gradients/Sum_grad/Maximum/y*
T0*1
_class'
%#loc:@train/gradients/Sum_grad/Shape*#
_output_shapes
:’’’’’’’’’
Ē
!train/gradients/Sum_grad/floordivFloorDivtrain/gradients/Sum_grad/Shape train/gradients/Sum_grad/Maximum*
T0*1
_class'
%#loc:@train/gradients/Sum_grad/Shape*
_output_shapes
:
¢
 train/gradients/Sum_grad/ReshapeReshapetrain/gradients/Neg_grad/Neg&train/gradients/Sum_grad/DynamicStitch*
T0*
Tshape0*
_output_shapes
:
®
train/gradients/Sum_grad/TileTile train/gradients/Sum_grad/Reshape!train/gradients/Sum_grad/floordiv*

Tmultiples0*
T0*'
_output_shapes
:’’’’’’’’’

l
train/gradients/mul_grad/ShapeShapeinputs/y_input*
T0*
out_type0*
_output_shapes
:
c
 train/gradients/mul_grad/Shape_1ShapeLog*
T0*
out_type0*
_output_shapes
:
Ę
.train/gradients/mul_grad/BroadcastGradientArgsBroadcastGradientArgstrain/gradients/mul_grad/Shape train/gradients/mul_grad/Shape_1*
T0*2
_output_shapes 
:’’’’’’’’’:’’’’’’’’’
y
train/gradients/mul_grad/mulMultrain/gradients/Sum_grad/TileLog*
T0*'
_output_shapes
:’’’’’’’’’

±
train/gradients/mul_grad/SumSumtrain/gradients/mul_grad/mul.train/gradients/mul_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
©
 train/gradients/mul_grad/ReshapeReshapetrain/gradients/mul_grad/Sumtrain/gradients/mul_grad/Shape*
T0*
Tshape0*'
_output_shapes
:’’’’’’’’’


train/gradients/mul_grad/mul_1Mulinputs/y_inputtrain/gradients/Sum_grad/Tile*
T0*'
_output_shapes
:’’’’’’’’’

·
train/gradients/mul_grad/Sum_1Sumtrain/gradients/mul_grad/mul_10train/gradients/mul_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
Æ
"train/gradients/mul_grad/Reshape_1Reshapetrain/gradients/mul_grad/Sum_1 train/gradients/mul_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:’’’’’’’’’

y
)train/gradients/mul_grad/tuple/group_depsNoOp!^train/gradients/mul_grad/Reshape#^train/gradients/mul_grad/Reshape_1
ņ
1train/gradients/mul_grad/tuple/control_dependencyIdentity train/gradients/mul_grad/Reshape*^train/gradients/mul_grad/tuple/group_deps*
T0*3
_class)
'%loc:@train/gradients/mul_grad/Reshape*'
_output_shapes
:’’’’’’’’’

ų
3train/gradients/mul_grad/tuple/control_dependency_1Identity"train/gradients/mul_grad/Reshape_1*^train/gradients/mul_grad/tuple/group_deps*
T0*5
_class+
)'loc:@train/gradients/mul_grad/Reshape_1*'
_output_shapes
:’’’’’’’’’

¢
#train/gradients/Log_grad/Reciprocal
ReciprocalSoftmax4^train/gradients/mul_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:’’’’’’’’’

Æ
train/gradients/Log_grad/mulMul3train/gradients/mul_grad/tuple/control_dependency_1#train/gradients/Log_grad/Reciprocal*
T0*'
_output_shapes
:’’’’’’’’’


 train/gradients/Softmax_grad/mulMultrain/gradients/Log_grad/mulSoftmax*
T0*'
_output_shapes
:’’’’’’’’’

|
2train/gradients/Softmax_grad/Sum/reduction_indicesConst*
valueB:*
dtype0*
_output_shapes
:
Č
 train/gradients/Softmax_grad/SumSum train/gradients/Softmax_grad/mul2train/gradients/Softmax_grad/Sum/reduction_indices*

Tidx0*
	keep_dims( *
T0*#
_output_shapes
:’’’’’’’’’
{
*train/gradients/Softmax_grad/Reshape/shapeConst*
valueB"’’’’   *
dtype0*
_output_shapes
:
½
$train/gradients/Softmax_grad/ReshapeReshape train/gradients/Softmax_grad/Sum*train/gradients/Softmax_grad/Reshape/shape*
T0*
Tshape0*'
_output_shapes
:’’’’’’’’’

 train/gradients/Softmax_grad/subSubtrain/gradients/Log_grad/mul$train/gradients/Softmax_grad/Reshape*
T0*'
_output_shapes
:’’’’’’’’’


"train/gradients/Softmax_grad/mul_1Mul train/gradients/Softmax_grad/subSoftmax*
T0*'
_output_shapes
:’’’’’’’’’

d
train/gradients/add_grad/ShapeShapeMatMul*
T0*
out_type0*
_output_shapes
:
j
 train/gradients/add_grad/Shape_1Const*
valueB:
*
dtype0*
_output_shapes
:
Ę
.train/gradients/add_grad/BroadcastGradientArgsBroadcastGradientArgstrain/gradients/add_grad/Shape train/gradients/add_grad/Shape_1*
T0*2
_output_shapes 
:’’’’’’’’’:’’’’’’’’’
·
train/gradients/add_grad/SumSum"train/gradients/Softmax_grad/mul_1.train/gradients/add_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
©
 train/gradients/add_grad/ReshapeReshapetrain/gradients/add_grad/Sumtrain/gradients/add_grad/Shape*
T0*
Tshape0*'
_output_shapes
:’’’’’’’’’

»
train/gradients/add_grad/Sum_1Sum"train/gradients/Softmax_grad/mul_10train/gradients/add_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
¢
"train/gradients/add_grad/Reshape_1Reshapetrain/gradients/add_grad/Sum_1 train/gradients/add_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:

y
)train/gradients/add_grad/tuple/group_depsNoOp!^train/gradients/add_grad/Reshape#^train/gradients/add_grad/Reshape_1
ņ
1train/gradients/add_grad/tuple/control_dependencyIdentity train/gradients/add_grad/Reshape*^train/gradients/add_grad/tuple/group_deps*
T0*3
_class)
'%loc:@train/gradients/add_grad/Reshape*'
_output_shapes
:’’’’’’’’’

ė
3train/gradients/add_grad/tuple/control_dependency_1Identity"train/gradients/add_grad/Reshape_1*^train/gradients/add_grad/tuple/group_deps*
T0*5
_class+
)'loc:@train/gradients/add_grad/Reshape_1*
_output_shapes
:

į
"train/gradients/MatMul_grad/MatMulMatMul1train/gradients/add_grad/tuple/control_dependency'full_connection_2/weights/Variable/read*
transpose_b(*
T0*(
_output_shapes
:’’’’’’’’’*
transpose_a( 
Š
$train/gradients/MatMul_grad/MatMul_1MatMulfull_connection_1/dropout/mul1train/gradients/add_grad/tuple/control_dependency*
transpose_b( *
T0*
_output_shapes
:	
*
transpose_a(

,train/gradients/MatMul_grad/tuple/group_depsNoOp#^train/gradients/MatMul_grad/MatMul%^train/gradients/MatMul_grad/MatMul_1
ż
4train/gradients/MatMul_grad/tuple/control_dependencyIdentity"train/gradients/MatMul_grad/MatMul-^train/gradients/MatMul_grad/tuple/group_deps*
T0*5
_class+
)'loc:@train/gradients/MatMul_grad/MatMul*(
_output_shapes
:’’’’’’’’’
ś
6train/gradients/MatMul_grad/tuple/control_dependency_1Identity$train/gradients/MatMul_grad/MatMul_1-^train/gradients/MatMul_grad/tuple/group_deps*
T0*7
_class-
+)loc:@train/gradients/MatMul_grad/MatMul_1*
_output_shapes
:	


8train/gradients/full_connection_1/dropout/mul_grad/ShapeShapefull_connection_1/dropout/div*
T0*
out_type0*
_output_shapes
:

:train/gradients/full_connection_1/dropout/mul_grad/Shape_1Shapefull_connection_1/dropout/Floor*
T0*
out_type0*
_output_shapes
:

Htrain/gradients/full_connection_1/dropout/mul_grad/BroadcastGradientArgsBroadcastGradientArgs8train/gradients/full_connection_1/dropout/mul_grad/Shape:train/gradients/full_connection_1/dropout/mul_grad/Shape_1*
T0*2
_output_shapes 
:’’’’’’’’’:’’’’’’’’’
Ē
6train/gradients/full_connection_1/dropout/mul_grad/mulMul4train/gradients/MatMul_grad/tuple/control_dependencyfull_connection_1/dropout/Floor*
T0*(
_output_shapes
:’’’’’’’’’
’
6train/gradients/full_connection_1/dropout/mul_grad/SumSum6train/gradients/full_connection_1/dropout/mul_grad/mulHtrain/gradients/full_connection_1/dropout/mul_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
ų
:train/gradients/full_connection_1/dropout/mul_grad/ReshapeReshape6train/gradients/full_connection_1/dropout/mul_grad/Sum8train/gradients/full_connection_1/dropout/mul_grad/Shape*
T0*
Tshape0*(
_output_shapes
:’’’’’’’’’
Ē
8train/gradients/full_connection_1/dropout/mul_grad/mul_1Mulfull_connection_1/dropout/div4train/gradients/MatMul_grad/tuple/control_dependency*
T0*(
_output_shapes
:’’’’’’’’’

8train/gradients/full_connection_1/dropout/mul_grad/Sum_1Sum8train/gradients/full_connection_1/dropout/mul_grad/mul_1Jtrain/gradients/full_connection_1/dropout/mul_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
ž
<train/gradients/full_connection_1/dropout/mul_grad/Reshape_1Reshape8train/gradients/full_connection_1/dropout/mul_grad/Sum_1:train/gradients/full_connection_1/dropout/mul_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:’’’’’’’’’
Ē
Ctrain/gradients/full_connection_1/dropout/mul_grad/tuple/group_depsNoOp;^train/gradients/full_connection_1/dropout/mul_grad/Reshape=^train/gradients/full_connection_1/dropout/mul_grad/Reshape_1
Ū
Ktrain/gradients/full_connection_1/dropout/mul_grad/tuple/control_dependencyIdentity:train/gradients/full_connection_1/dropout/mul_grad/ReshapeD^train/gradients/full_connection_1/dropout/mul_grad/tuple/group_deps*
T0*M
_classC
A?loc:@train/gradients/full_connection_1/dropout/mul_grad/Reshape*(
_output_shapes
:’’’’’’’’’
į
Mtrain/gradients/full_connection_1/dropout/mul_grad/tuple/control_dependency_1Identity<train/gradients/full_connection_1/dropout/mul_grad/Reshape_1D^train/gradients/full_connection_1/dropout/mul_grad/tuple/group_deps*
T0*O
_classE
CAloc:@train/gradients/full_connection_1/dropout/mul_grad/Reshape_1*(
_output_shapes
:’’’’’’’’’

8train/gradients/full_connection_1/dropout/div_grad/ShapeShapefull_connection_1/Relu*
T0*
out_type0*
_output_shapes
:
}
:train/gradients/full_connection_1/dropout/div_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 

Htrain/gradients/full_connection_1/dropout/div_grad/BroadcastGradientArgsBroadcastGradientArgs8train/gradients/full_connection_1/dropout/div_grad/Shape:train/gradients/full_connection_1/dropout/div_grad/Shape_1*
T0*2
_output_shapes 
:’’’’’’’’’:’’’’’’’’’
ź
:train/gradients/full_connection_1/dropout/div_grad/RealDivRealDivKtrain/gradients/full_connection_1/dropout/mul_grad/tuple/control_dependency#full_connection_1/dropout/keep_prob*
T0*(
_output_shapes
:’’’’’’’’’

6train/gradients/full_connection_1/dropout/div_grad/SumSum:train/gradients/full_connection_1/dropout/div_grad/RealDivHtrain/gradients/full_connection_1/dropout/div_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
ų
:train/gradients/full_connection_1/dropout/div_grad/ReshapeReshape6train/gradients/full_connection_1/dropout/div_grad/Sum8train/gradients/full_connection_1/dropout/div_grad/Shape*
T0*
Tshape0*(
_output_shapes
:’’’’’’’’’

6train/gradients/full_connection_1/dropout/div_grad/NegNegfull_connection_1/Relu*
T0*(
_output_shapes
:’’’’’’’’’
×
<train/gradients/full_connection_1/dropout/div_grad/RealDiv_1RealDiv6train/gradients/full_connection_1/dropout/div_grad/Neg#full_connection_1/dropout/keep_prob*
T0*(
_output_shapes
:’’’’’’’’’
Ż
<train/gradients/full_connection_1/dropout/div_grad/RealDiv_2RealDiv<train/gradients/full_connection_1/dropout/div_grad/RealDiv_1#full_connection_1/dropout/keep_prob*
T0*(
_output_shapes
:’’’’’’’’’
ū
6train/gradients/full_connection_1/dropout/div_grad/mulMulKtrain/gradients/full_connection_1/dropout/mul_grad/tuple/control_dependency<train/gradients/full_connection_1/dropout/div_grad/RealDiv_2*
T0*(
_output_shapes
:’’’’’’’’’

8train/gradients/full_connection_1/dropout/div_grad/Sum_1Sum6train/gradients/full_connection_1/dropout/div_grad/mulJtrain/gradients/full_connection_1/dropout/div_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
ģ
<train/gradients/full_connection_1/dropout/div_grad/Reshape_1Reshape8train/gradients/full_connection_1/dropout/div_grad/Sum_1:train/gradients/full_connection_1/dropout/div_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
Ē
Ctrain/gradients/full_connection_1/dropout/div_grad/tuple/group_depsNoOp;^train/gradients/full_connection_1/dropout/div_grad/Reshape=^train/gradients/full_connection_1/dropout/div_grad/Reshape_1
Ū
Ktrain/gradients/full_connection_1/dropout/div_grad/tuple/control_dependencyIdentity:train/gradients/full_connection_1/dropout/div_grad/ReshapeD^train/gradients/full_connection_1/dropout/div_grad/tuple/group_deps*
T0*M
_classC
A?loc:@train/gradients/full_connection_1/dropout/div_grad/Reshape*(
_output_shapes
:’’’’’’’’’
Ļ
Mtrain/gradients/full_connection_1/dropout/div_grad/tuple/control_dependency_1Identity<train/gradients/full_connection_1/dropout/div_grad/Reshape_1D^train/gradients/full_connection_1/dropout/div_grad/tuple/group_deps*
T0*O
_classE
CAloc:@train/gradients/full_connection_1/dropout/div_grad/Reshape_1*
_output_shapes
: 
Ų
4train/gradients/full_connection_1/Relu_grad/ReluGradReluGradKtrain/gradients/full_connection_1/dropout/div_grad/tuple/control_dependencyfull_connection_1/Relu*
T0*(
_output_shapes
:’’’’’’’’’
ö
4train/gradients/full_connection_1/MatMul_grad/MatMulMatMul4train/gradients/full_connection_1/Relu_grad/ReluGrad'full_connection_1/weights/Variable/read*
transpose_b(*
T0*(
_output_shapes
:’’’’’’’’’Ą*
transpose_a( 
ā
6train/gradients/full_connection_1/MatMul_grad/MatMul_1MatMulfull_connection_1/Reshape4train/gradients/full_connection_1/Relu_grad/ReluGrad*
transpose_b( *
T0* 
_output_shapes
:
Ą*
transpose_a(
¶
>train/gradients/full_connection_1/MatMul_grad/tuple/group_depsNoOp5^train/gradients/full_connection_1/MatMul_grad/MatMul7^train/gradients/full_connection_1/MatMul_grad/MatMul_1
Å
Ftrain/gradients/full_connection_1/MatMul_grad/tuple/control_dependencyIdentity4train/gradients/full_connection_1/MatMul_grad/MatMul?^train/gradients/full_connection_1/MatMul_grad/tuple/group_deps*
T0*G
_class=
;9loc:@train/gradients/full_connection_1/MatMul_grad/MatMul*(
_output_shapes
:’’’’’’’’’Ą
Ć
Htrain/gradients/full_connection_1/MatMul_grad/tuple/control_dependency_1Identity6train/gradients/full_connection_1/MatMul_grad/MatMul_1?^train/gradients/full_connection_1/MatMul_grad/tuple/group_deps*
T0*I
_class?
=;loc:@train/gradients/full_connection_1/MatMul_grad/MatMul_1* 
_output_shapes
:
Ą

4train/gradients/full_connection_1/Reshape_grad/ShapeShapehidden_layer_2/MaxPool*
T0*
out_type0*
_output_shapes
:

6train/gradients/full_connection_1/Reshape_grad/ReshapeReshapeFtrain/gradients/full_connection_1/MatMul_grad/tuple/control_dependency4train/gradients/full_connection_1/Reshape_grad/Shape*
T0*
Tshape0*/
_output_shapes
:’’’’’’’’’@
·
7train/gradients/hidden_layer_2/MaxPool_grad/MaxPoolGradMaxPoolGradhidden_layer_2/Reluhidden_layer_2/MaxPool6train/gradients/full_connection_1/Reshape_grad/Reshape*
T0*
strides
*
data_formatNHWC*
ksize
*
paddingSAME*/
_output_shapes
:’’’’’’’’’@
Å
1train/gradients/hidden_layer_2/Relu_grad/ReluGradReluGrad7train/gradients/hidden_layer_2/MaxPool_grad/MaxPoolGradhidden_layer_2/Relu*
T0*/
_output_shapes
:’’’’’’’’’@

-train/gradients/hidden_layer_2/add_grad/ShapeShapehidden_layer_2/Conv2D*
T0*
out_type0*
_output_shapes
:
y
/train/gradients/hidden_layer_2/add_grad/Shape_1Const*
valueB:@*
dtype0*
_output_shapes
:
ó
=train/gradients/hidden_layer_2/add_grad/BroadcastGradientArgsBroadcastGradientArgs-train/gradients/hidden_layer_2/add_grad/Shape/train/gradients/hidden_layer_2/add_grad/Shape_1*
T0*2
_output_shapes 
:’’’’’’’’’:’’’’’’’’’
ä
+train/gradients/hidden_layer_2/add_grad/SumSum1train/gradients/hidden_layer_2/Relu_grad/ReluGrad=train/gradients/hidden_layer_2/add_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
Ž
/train/gradients/hidden_layer_2/add_grad/ReshapeReshape+train/gradients/hidden_layer_2/add_grad/Sum-train/gradients/hidden_layer_2/add_grad/Shape*
T0*
Tshape0*/
_output_shapes
:’’’’’’’’’@
č
-train/gradients/hidden_layer_2/add_grad/Sum_1Sum1train/gradients/hidden_layer_2/Relu_grad/ReluGrad?train/gradients/hidden_layer_2/add_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
Ļ
1train/gradients/hidden_layer_2/add_grad/Reshape_1Reshape-train/gradients/hidden_layer_2/add_grad/Sum_1/train/gradients/hidden_layer_2/add_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:@
¦
8train/gradients/hidden_layer_2/add_grad/tuple/group_depsNoOp0^train/gradients/hidden_layer_2/add_grad/Reshape2^train/gradients/hidden_layer_2/add_grad/Reshape_1
¶
@train/gradients/hidden_layer_2/add_grad/tuple/control_dependencyIdentity/train/gradients/hidden_layer_2/add_grad/Reshape9^train/gradients/hidden_layer_2/add_grad/tuple/group_deps*
T0*B
_class8
64loc:@train/gradients/hidden_layer_2/add_grad/Reshape*/
_output_shapes
:’’’’’’’’’@
§
Btrain/gradients/hidden_layer_2/add_grad/tuple/control_dependency_1Identity1train/gradients/hidden_layer_2/add_grad/Reshape_19^train/gradients/hidden_layer_2/add_grad/tuple/group_deps*
T0*D
_class:
86loc:@train/gradients/hidden_layer_2/add_grad/Reshape_1*
_output_shapes
:@
½
1train/gradients/hidden_layer_2/Conv2D_grad/ShapeNShapeNhidden_layer_1/MaxPool$hidden_layer_2/weights/Variable/read*
T0*
out_type0*
N* 
_output_shapes
::

0train/gradients/hidden_layer_2/Conv2D_grad/ConstConst*%
valueB"          @   *
dtype0*
_output_shapes
:
³
>train/gradients/hidden_layer_2/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput1train/gradients/hidden_layer_2/Conv2D_grad/ShapeN$hidden_layer_2/weights/Variable/read@train/gradients/hidden_layer_2/add_grad/tuple/control_dependency*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*J
_output_shapes8
6:4’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’

?train/gradients/hidden_layer_2/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterhidden_layer_1/MaxPool0train/gradients/hidden_layer_2/Conv2D_grad/Const@train/gradients/hidden_layer_2/add_grad/tuple/control_dependency*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*&
_output_shapes
: @
Ę
;train/gradients/hidden_layer_2/Conv2D_grad/tuple/group_depsNoOp?^train/gradients/hidden_layer_2/Conv2D_grad/Conv2DBackpropInput@^train/gradients/hidden_layer_2/Conv2D_grad/Conv2DBackpropFilter
Ś
Ctrain/gradients/hidden_layer_2/Conv2D_grad/tuple/control_dependencyIdentity>train/gradients/hidden_layer_2/Conv2D_grad/Conv2DBackpropInput<^train/gradients/hidden_layer_2/Conv2D_grad/tuple/group_deps*
T0*Q
_classG
ECloc:@train/gradients/hidden_layer_2/Conv2D_grad/Conv2DBackpropInput*/
_output_shapes
:’’’’’’’’’ 
Õ
Etrain/gradients/hidden_layer_2/Conv2D_grad/tuple/control_dependency_1Identity?train/gradients/hidden_layer_2/Conv2D_grad/Conv2DBackpropFilter<^train/gradients/hidden_layer_2/Conv2D_grad/tuple/group_deps*
T0*R
_classH
FDloc:@train/gradients/hidden_layer_2/Conv2D_grad/Conv2DBackpropFilter*&
_output_shapes
: @
Ä
7train/gradients/hidden_layer_1/MaxPool_grad/MaxPoolGradMaxPoolGradhidden_layer_1/Reluhidden_layer_1/MaxPoolCtrain/gradients/hidden_layer_2/Conv2D_grad/tuple/control_dependency*
T0*
strides
*
data_formatNHWC*
ksize
*
paddingSAME*/
_output_shapes
:’’’’’’’’’ 
Å
1train/gradients/hidden_layer_1/Relu_grad/ReluGradReluGrad7train/gradients/hidden_layer_1/MaxPool_grad/MaxPoolGradhidden_layer_1/Relu*
T0*/
_output_shapes
:’’’’’’’’’ 

-train/gradients/hidden_layer_1/add_grad/ShapeShapehidden_layer_1/Conv2D*
T0*
out_type0*
_output_shapes
:
y
/train/gradients/hidden_layer_1/add_grad/Shape_1Const*
valueB: *
dtype0*
_output_shapes
:
ó
=train/gradients/hidden_layer_1/add_grad/BroadcastGradientArgsBroadcastGradientArgs-train/gradients/hidden_layer_1/add_grad/Shape/train/gradients/hidden_layer_1/add_grad/Shape_1*
T0*2
_output_shapes 
:’’’’’’’’’:’’’’’’’’’
ä
+train/gradients/hidden_layer_1/add_grad/SumSum1train/gradients/hidden_layer_1/Relu_grad/ReluGrad=train/gradients/hidden_layer_1/add_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
Ž
/train/gradients/hidden_layer_1/add_grad/ReshapeReshape+train/gradients/hidden_layer_1/add_grad/Sum-train/gradients/hidden_layer_1/add_grad/Shape*
T0*
Tshape0*/
_output_shapes
:’’’’’’’’’ 
č
-train/gradients/hidden_layer_1/add_grad/Sum_1Sum1train/gradients/hidden_layer_1/Relu_grad/ReluGrad?train/gradients/hidden_layer_1/add_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
Ļ
1train/gradients/hidden_layer_1/add_grad/Reshape_1Reshape-train/gradients/hidden_layer_1/add_grad/Sum_1/train/gradients/hidden_layer_1/add_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
¦
8train/gradients/hidden_layer_1/add_grad/tuple/group_depsNoOp0^train/gradients/hidden_layer_1/add_grad/Reshape2^train/gradients/hidden_layer_1/add_grad/Reshape_1
¶
@train/gradients/hidden_layer_1/add_grad/tuple/control_dependencyIdentity/train/gradients/hidden_layer_1/add_grad/Reshape9^train/gradients/hidden_layer_1/add_grad/tuple/group_deps*
T0*B
_class8
64loc:@train/gradients/hidden_layer_1/add_grad/Reshape*/
_output_shapes
:’’’’’’’’’ 
§
Btrain/gradients/hidden_layer_1/add_grad/tuple/control_dependency_1Identity1train/gradients/hidden_layer_1/add_grad/Reshape_19^train/gradients/hidden_layer_1/add_grad/tuple/group_deps*
T0*D
_class:
86loc:@train/gradients/hidden_layer_1/add_grad/Reshape_1*
_output_shapes
: 
²
1train/gradients/hidden_layer_1/Conv2D_grad/ShapeNShapeNimage_input$hidden_layer_1/weights/Variable/read*
T0*
out_type0*
N* 
_output_shapes
::

0train/gradients/hidden_layer_1/Conv2D_grad/ConstConst*%
valueB"             *
dtype0*
_output_shapes
:
³
>train/gradients/hidden_layer_1/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput1train/gradients/hidden_layer_1/Conv2D_grad/ShapeN$hidden_layer_1/weights/Variable/read@train/gradients/hidden_layer_1/add_grad/tuple/control_dependency*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*J
_output_shapes8
6:4’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’
÷
?train/gradients/hidden_layer_1/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterimage_input0train/gradients/hidden_layer_1/Conv2D_grad/Const@train/gradients/hidden_layer_1/add_grad/tuple/control_dependency*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*&
_output_shapes
: 
Ę
;train/gradients/hidden_layer_1/Conv2D_grad/tuple/group_depsNoOp?^train/gradients/hidden_layer_1/Conv2D_grad/Conv2DBackpropInput@^train/gradients/hidden_layer_1/Conv2D_grad/Conv2DBackpropFilter
Ś
Ctrain/gradients/hidden_layer_1/Conv2D_grad/tuple/control_dependencyIdentity>train/gradients/hidden_layer_1/Conv2D_grad/Conv2DBackpropInput<^train/gradients/hidden_layer_1/Conv2D_grad/tuple/group_deps*
T0*Q
_classG
ECloc:@train/gradients/hidden_layer_1/Conv2D_grad/Conv2DBackpropInput*/
_output_shapes
:’’’’’’’’’
Õ
Etrain/gradients/hidden_layer_1/Conv2D_grad/tuple/control_dependency_1Identity?train/gradients/hidden_layer_1/Conv2D_grad/Conv2DBackpropFilter<^train/gradients/hidden_layer_1/Conv2D_grad/tuple/group_deps*
T0*R
_classH
FDloc:@train/gradients/hidden_layer_1/Conv2D_grad/Conv2DBackpropFilter*&
_output_shapes
: 

train/beta1_power/initial_valueConst*
valueB
 *fff?*5
_class+
)'loc:@full_connection_1/weights/Variable*
dtype0*
_output_shapes
: 
¬
train/beta1_power
VariableV2*
shared_name *5
_class+
)'loc:@full_connection_1/weights/Variable*
	container *
shape: *
dtype0*
_output_shapes
: 
×
train/beta1_power/AssignAssigntrain/beta1_powertrain/beta1_power/initial_value*
use_locking(*
T0*5
_class+
)'loc:@full_connection_1/weights/Variable*
validate_shape(*
_output_shapes
: 

train/beta1_power/readIdentitytrain/beta1_power*
T0*5
_class+
)'loc:@full_connection_1/weights/Variable*
_output_shapes
: 

train/beta2_power/initial_valueConst*
valueB
 *w¾?*5
_class+
)'loc:@full_connection_1/weights/Variable*
dtype0*
_output_shapes
: 
¬
train/beta2_power
VariableV2*
shared_name *5
_class+
)'loc:@full_connection_1/weights/Variable*
	container *
shape: *
dtype0*
_output_shapes
: 
×
train/beta2_power/AssignAssigntrain/beta2_powertrain/beta2_power/initial_value*
use_locking(*
T0*5
_class+
)'loc:@full_connection_1/weights/Variable*
validate_shape(*
_output_shapes
: 

train/beta2_power/readIdentitytrain/beta2_power*
T0*5
_class+
)'loc:@full_connection_1/weights/Variable*
_output_shapes
: 
Ļ
6hidden_layer_1/weights/Variable/Adam/Initializer/zerosConst*%
valueB *    *2
_class(
&$loc:@hidden_layer_1/weights/Variable*
dtype0*&
_output_shapes
: 
Ü
$hidden_layer_1/weights/Variable/Adam
VariableV2*
shared_name *2
_class(
&$loc:@hidden_layer_1/weights/Variable*
	container *
shape: *
dtype0*&
_output_shapes
: 
”
+hidden_layer_1/weights/Variable/Adam/AssignAssign$hidden_layer_1/weights/Variable/Adam6hidden_layer_1/weights/Variable/Adam/Initializer/zeros*
use_locking(*
T0*2
_class(
&$loc:@hidden_layer_1/weights/Variable*
validate_shape(*&
_output_shapes
: 
Ą
)hidden_layer_1/weights/Variable/Adam/readIdentity$hidden_layer_1/weights/Variable/Adam*
T0*2
_class(
&$loc:@hidden_layer_1/weights/Variable*&
_output_shapes
: 
Ń
8hidden_layer_1/weights/Variable/Adam_1/Initializer/zerosConst*%
valueB *    *2
_class(
&$loc:@hidden_layer_1/weights/Variable*
dtype0*&
_output_shapes
: 
Ž
&hidden_layer_1/weights/Variable/Adam_1
VariableV2*
shared_name *2
_class(
&$loc:@hidden_layer_1/weights/Variable*
	container *
shape: *
dtype0*&
_output_shapes
: 
§
-hidden_layer_1/weights/Variable/Adam_1/AssignAssign&hidden_layer_1/weights/Variable/Adam_18hidden_layer_1/weights/Variable/Adam_1/Initializer/zeros*
use_locking(*
T0*2
_class(
&$loc:@hidden_layer_1/weights/Variable*
validate_shape(*&
_output_shapes
: 
Ä
+hidden_layer_1/weights/Variable/Adam_1/readIdentity&hidden_layer_1/weights/Variable/Adam_1*
T0*2
_class(
&$loc:@hidden_layer_1/weights/Variable*&
_output_shapes
: 
µ
5hidden_layer_1/biases/Variable/Adam/Initializer/zerosConst*
valueB *    *1
_class'
%#loc:@hidden_layer_1/biases/Variable*
dtype0*
_output_shapes
: 
Ā
#hidden_layer_1/biases/Variable/Adam
VariableV2*
shared_name *1
_class'
%#loc:@hidden_layer_1/biases/Variable*
	container *
shape: *
dtype0*
_output_shapes
: 

*hidden_layer_1/biases/Variable/Adam/AssignAssign#hidden_layer_1/biases/Variable/Adam5hidden_layer_1/biases/Variable/Adam/Initializer/zeros*
use_locking(*
T0*1
_class'
%#loc:@hidden_layer_1/biases/Variable*
validate_shape(*
_output_shapes
: 
±
(hidden_layer_1/biases/Variable/Adam/readIdentity#hidden_layer_1/biases/Variable/Adam*
T0*1
_class'
%#loc:@hidden_layer_1/biases/Variable*
_output_shapes
: 
·
7hidden_layer_1/biases/Variable/Adam_1/Initializer/zerosConst*
valueB *    *1
_class'
%#loc:@hidden_layer_1/biases/Variable*
dtype0*
_output_shapes
: 
Ä
%hidden_layer_1/biases/Variable/Adam_1
VariableV2*
shared_name *1
_class'
%#loc:@hidden_layer_1/biases/Variable*
	container *
shape: *
dtype0*
_output_shapes
: 

,hidden_layer_1/biases/Variable/Adam_1/AssignAssign%hidden_layer_1/biases/Variable/Adam_17hidden_layer_1/biases/Variable/Adam_1/Initializer/zeros*
use_locking(*
T0*1
_class'
%#loc:@hidden_layer_1/biases/Variable*
validate_shape(*
_output_shapes
: 
µ
*hidden_layer_1/biases/Variable/Adam_1/readIdentity%hidden_layer_1/biases/Variable/Adam_1*
T0*1
_class'
%#loc:@hidden_layer_1/biases/Variable*
_output_shapes
: 
Ļ
6hidden_layer_2/weights/Variable/Adam/Initializer/zerosConst*%
valueB @*    *2
_class(
&$loc:@hidden_layer_2/weights/Variable*
dtype0*&
_output_shapes
: @
Ü
$hidden_layer_2/weights/Variable/Adam
VariableV2*
shared_name *2
_class(
&$loc:@hidden_layer_2/weights/Variable*
	container *
shape: @*
dtype0*&
_output_shapes
: @
”
+hidden_layer_2/weights/Variable/Adam/AssignAssign$hidden_layer_2/weights/Variable/Adam6hidden_layer_2/weights/Variable/Adam/Initializer/zeros*
use_locking(*
T0*2
_class(
&$loc:@hidden_layer_2/weights/Variable*
validate_shape(*&
_output_shapes
: @
Ą
)hidden_layer_2/weights/Variable/Adam/readIdentity$hidden_layer_2/weights/Variable/Adam*
T0*2
_class(
&$loc:@hidden_layer_2/weights/Variable*&
_output_shapes
: @
Ń
8hidden_layer_2/weights/Variable/Adam_1/Initializer/zerosConst*%
valueB @*    *2
_class(
&$loc:@hidden_layer_2/weights/Variable*
dtype0*&
_output_shapes
: @
Ž
&hidden_layer_2/weights/Variable/Adam_1
VariableV2*
shared_name *2
_class(
&$loc:@hidden_layer_2/weights/Variable*
	container *
shape: @*
dtype0*&
_output_shapes
: @
§
-hidden_layer_2/weights/Variable/Adam_1/AssignAssign&hidden_layer_2/weights/Variable/Adam_18hidden_layer_2/weights/Variable/Adam_1/Initializer/zeros*
use_locking(*
T0*2
_class(
&$loc:@hidden_layer_2/weights/Variable*
validate_shape(*&
_output_shapes
: @
Ä
+hidden_layer_2/weights/Variable/Adam_1/readIdentity&hidden_layer_2/weights/Variable/Adam_1*
T0*2
_class(
&$loc:@hidden_layer_2/weights/Variable*&
_output_shapes
: @
µ
5hidden_layer_2/biases/Variable/Adam/Initializer/zerosConst*
valueB@*    *1
_class'
%#loc:@hidden_layer_2/biases/Variable*
dtype0*
_output_shapes
:@
Ā
#hidden_layer_2/biases/Variable/Adam
VariableV2*
shared_name *1
_class'
%#loc:@hidden_layer_2/biases/Variable*
	container *
shape:@*
dtype0*
_output_shapes
:@

*hidden_layer_2/biases/Variable/Adam/AssignAssign#hidden_layer_2/biases/Variable/Adam5hidden_layer_2/biases/Variable/Adam/Initializer/zeros*
use_locking(*
T0*1
_class'
%#loc:@hidden_layer_2/biases/Variable*
validate_shape(*
_output_shapes
:@
±
(hidden_layer_2/biases/Variable/Adam/readIdentity#hidden_layer_2/biases/Variable/Adam*
T0*1
_class'
%#loc:@hidden_layer_2/biases/Variable*
_output_shapes
:@
·
7hidden_layer_2/biases/Variable/Adam_1/Initializer/zerosConst*
valueB@*    *1
_class'
%#loc:@hidden_layer_2/biases/Variable*
dtype0*
_output_shapes
:@
Ä
%hidden_layer_2/biases/Variable/Adam_1
VariableV2*
shared_name *1
_class'
%#loc:@hidden_layer_2/biases/Variable*
	container *
shape:@*
dtype0*
_output_shapes
:@

,hidden_layer_2/biases/Variable/Adam_1/AssignAssign%hidden_layer_2/biases/Variable/Adam_17hidden_layer_2/biases/Variable/Adam_1/Initializer/zeros*
use_locking(*
T0*1
_class'
%#loc:@hidden_layer_2/biases/Variable*
validate_shape(*
_output_shapes
:@
µ
*hidden_layer_2/biases/Variable/Adam_1/readIdentity%hidden_layer_2/biases/Variable/Adam_1*
T0*1
_class'
%#loc:@hidden_layer_2/biases/Variable*
_output_shapes
:@
É
9full_connection_1/weights/Variable/Adam/Initializer/zerosConst*
valueB
Ą*    *5
_class+
)'loc:@full_connection_1/weights/Variable*
dtype0* 
_output_shapes
:
Ą
Ö
'full_connection_1/weights/Variable/Adam
VariableV2*
shared_name *5
_class+
)'loc:@full_connection_1/weights/Variable*
	container *
shape:
Ą*
dtype0* 
_output_shapes
:
Ą
§
.full_connection_1/weights/Variable/Adam/AssignAssign'full_connection_1/weights/Variable/Adam9full_connection_1/weights/Variable/Adam/Initializer/zeros*
use_locking(*
T0*5
_class+
)'loc:@full_connection_1/weights/Variable*
validate_shape(* 
_output_shapes
:
Ą
Ć
,full_connection_1/weights/Variable/Adam/readIdentity'full_connection_1/weights/Variable/Adam*
T0*5
_class+
)'loc:@full_connection_1/weights/Variable* 
_output_shapes
:
Ą
Ė
;full_connection_1/weights/Variable/Adam_1/Initializer/zerosConst*
valueB
Ą*    *5
_class+
)'loc:@full_connection_1/weights/Variable*
dtype0* 
_output_shapes
:
Ą
Ų
)full_connection_1/weights/Variable/Adam_1
VariableV2*
shared_name *5
_class+
)'loc:@full_connection_1/weights/Variable*
	container *
shape:
Ą*
dtype0* 
_output_shapes
:
Ą
­
0full_connection_1/weights/Variable/Adam_1/AssignAssign)full_connection_1/weights/Variable/Adam_1;full_connection_1/weights/Variable/Adam_1/Initializer/zeros*
use_locking(*
T0*5
_class+
)'loc:@full_connection_1/weights/Variable*
validate_shape(* 
_output_shapes
:
Ą
Ē
.full_connection_1/weights/Variable/Adam_1/readIdentity)full_connection_1/weights/Variable/Adam_1*
T0*5
_class+
)'loc:@full_connection_1/weights/Variable* 
_output_shapes
:
Ą
Ē
9full_connection_2/weights/Variable/Adam/Initializer/zerosConst*
valueB	
*    *5
_class+
)'loc:@full_connection_2/weights/Variable*
dtype0*
_output_shapes
:	

Ō
'full_connection_2/weights/Variable/Adam
VariableV2*
shared_name *5
_class+
)'loc:@full_connection_2/weights/Variable*
	container *
shape:	
*
dtype0*
_output_shapes
:	

¦
.full_connection_2/weights/Variable/Adam/AssignAssign'full_connection_2/weights/Variable/Adam9full_connection_2/weights/Variable/Adam/Initializer/zeros*
use_locking(*
T0*5
_class+
)'loc:@full_connection_2/weights/Variable*
validate_shape(*
_output_shapes
:	

Ā
,full_connection_2/weights/Variable/Adam/readIdentity'full_connection_2/weights/Variable/Adam*
T0*5
_class+
)'loc:@full_connection_2/weights/Variable*
_output_shapes
:	

É
;full_connection_2/weights/Variable/Adam_1/Initializer/zerosConst*
valueB	
*    *5
_class+
)'loc:@full_connection_2/weights/Variable*
dtype0*
_output_shapes
:	

Ö
)full_connection_2/weights/Variable/Adam_1
VariableV2*
shared_name *5
_class+
)'loc:@full_connection_2/weights/Variable*
	container *
shape:	
*
dtype0*
_output_shapes
:	

¬
0full_connection_2/weights/Variable/Adam_1/AssignAssign)full_connection_2/weights/Variable/Adam_1;full_connection_2/weights/Variable/Adam_1/Initializer/zeros*
use_locking(*
T0*5
_class+
)'loc:@full_connection_2/weights/Variable*
validate_shape(*
_output_shapes
:	

Ę
.full_connection_2/weights/Variable/Adam_1/readIdentity)full_connection_2/weights/Variable/Adam_1*
T0*5
_class+
)'loc:@full_connection_2/weights/Variable*
_output_shapes
:	

»
8full_connection_2/biases/Variable/Adam/Initializer/zerosConst*
valueB
*    *4
_class*
(&loc:@full_connection_2/biases/Variable*
dtype0*
_output_shapes
:

Č
&full_connection_2/biases/Variable/Adam
VariableV2*
shared_name *4
_class*
(&loc:@full_connection_2/biases/Variable*
	container *
shape:
*
dtype0*
_output_shapes
:


-full_connection_2/biases/Variable/Adam/AssignAssign&full_connection_2/biases/Variable/Adam8full_connection_2/biases/Variable/Adam/Initializer/zeros*
use_locking(*
T0*4
_class*
(&loc:@full_connection_2/biases/Variable*
validate_shape(*
_output_shapes
:

ŗ
+full_connection_2/biases/Variable/Adam/readIdentity&full_connection_2/biases/Variable/Adam*
T0*4
_class*
(&loc:@full_connection_2/biases/Variable*
_output_shapes
:

½
:full_connection_2/biases/Variable/Adam_1/Initializer/zerosConst*
valueB
*    *4
_class*
(&loc:@full_connection_2/biases/Variable*
dtype0*
_output_shapes
:

Ź
(full_connection_2/biases/Variable/Adam_1
VariableV2*
shared_name *4
_class*
(&loc:@full_connection_2/biases/Variable*
	container *
shape:
*
dtype0*
_output_shapes
:

£
/full_connection_2/biases/Variable/Adam_1/AssignAssign(full_connection_2/biases/Variable/Adam_1:full_connection_2/biases/Variable/Adam_1/Initializer/zeros*
use_locking(*
T0*4
_class*
(&loc:@full_connection_2/biases/Variable*
validate_shape(*
_output_shapes
:

¾
-full_connection_2/biases/Variable/Adam_1/readIdentity(full_connection_2/biases/Variable/Adam_1*
T0*4
_class*
(&loc:@full_connection_2/biases/Variable*
_output_shapes
:

]
train/Adam/learning_rateConst*
valueB
 *·Ń8*
dtype0*
_output_shapes
: 
U
train/Adam/beta1Const*
valueB
 *fff?*
dtype0*
_output_shapes
: 
U
train/Adam/beta2Const*
valueB
 *w¾?*
dtype0*
_output_shapes
: 
W
train/Adam/epsilonConst*
valueB
 *wĢ+2*
dtype0*
_output_shapes
: 

;train/Adam/update_hidden_layer_1/weights/Variable/ApplyAdam	ApplyAdamhidden_layer_1/weights/Variable$hidden_layer_1/weights/Variable/Adam&hidden_layer_1/weights/Variable/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilonEtrain/gradients/hidden_layer_1/Conv2D_grad/tuple/control_dependency_1*
use_locking( *
T0*2
_class(
&$loc:@hidden_layer_1/weights/Variable*
use_nesterov( *&
_output_shapes
: 
ų
:train/Adam/update_hidden_layer_1/biases/Variable/ApplyAdam	ApplyAdamhidden_layer_1/biases/Variable#hidden_layer_1/biases/Variable/Adam%hidden_layer_1/biases/Variable/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilonBtrain/gradients/hidden_layer_1/add_grad/tuple/control_dependency_1*
use_locking( *
T0*1
_class'
%#loc:@hidden_layer_1/biases/Variable*
use_nesterov( *
_output_shapes
: 

;train/Adam/update_hidden_layer_2/weights/Variable/ApplyAdam	ApplyAdamhidden_layer_2/weights/Variable$hidden_layer_2/weights/Variable/Adam&hidden_layer_2/weights/Variable/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilonEtrain/gradients/hidden_layer_2/Conv2D_grad/tuple/control_dependency_1*
use_locking( *
T0*2
_class(
&$loc:@hidden_layer_2/weights/Variable*
use_nesterov( *&
_output_shapes
: @
ų
:train/Adam/update_hidden_layer_2/biases/Variable/ApplyAdam	ApplyAdamhidden_layer_2/biases/Variable#hidden_layer_2/biases/Variable/Adam%hidden_layer_2/biases/Variable/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilonBtrain/gradients/hidden_layer_2/add_grad/tuple/control_dependency_1*
use_locking( *
T0*1
_class'
%#loc:@hidden_layer_2/biases/Variable*
use_nesterov( *
_output_shapes
:@

>train/Adam/update_full_connection_1/weights/Variable/ApplyAdam	ApplyAdam"full_connection_1/weights/Variable'full_connection_1/weights/Variable/Adam)full_connection_1/weights/Variable/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilonHtrain/gradients/full_connection_1/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*5
_class+
)'loc:@full_connection_1/weights/Variable*
use_nesterov( * 
_output_shapes
:
Ą

>train/Adam/update_full_connection_2/weights/Variable/ApplyAdam	ApplyAdam"full_connection_2/weights/Variable'full_connection_2/weights/Variable/Adam)full_connection_2/weights/Variable/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilon6train/gradients/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*5
_class+
)'loc:@full_connection_2/weights/Variable*
use_nesterov( *
_output_shapes
:	

ų
=train/Adam/update_full_connection_2/biases/Variable/ApplyAdam	ApplyAdam!full_connection_2/biases/Variable&full_connection_2/biases/Variable/Adam(full_connection_2/biases/Variable/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilon3train/gradients/add_grad/tuple/control_dependency_1*
use_locking( *
T0*4
_class*
(&loc:@full_connection_2/biases/Variable*
use_nesterov( *
_output_shapes
:

Ļ
train/Adam/mulMultrain/beta1_power/readtrain/Adam/beta1<^train/Adam/update_hidden_layer_1/weights/Variable/ApplyAdam;^train/Adam/update_hidden_layer_1/biases/Variable/ApplyAdam<^train/Adam/update_hidden_layer_2/weights/Variable/ApplyAdam;^train/Adam/update_hidden_layer_2/biases/Variable/ApplyAdam?^train/Adam/update_full_connection_1/weights/Variable/ApplyAdam?^train/Adam/update_full_connection_2/weights/Variable/ApplyAdam>^train/Adam/update_full_connection_2/biases/Variable/ApplyAdam*
T0*5
_class+
)'loc:@full_connection_1/weights/Variable*
_output_shapes
: 
æ
train/Adam/AssignAssigntrain/beta1_powertrain/Adam/mul*
use_locking( *
T0*5
_class+
)'loc:@full_connection_1/weights/Variable*
validate_shape(*
_output_shapes
: 
Ń
train/Adam/mul_1Multrain/beta2_power/readtrain/Adam/beta2<^train/Adam/update_hidden_layer_1/weights/Variable/ApplyAdam;^train/Adam/update_hidden_layer_1/biases/Variable/ApplyAdam<^train/Adam/update_hidden_layer_2/weights/Variable/ApplyAdam;^train/Adam/update_hidden_layer_2/biases/Variable/ApplyAdam?^train/Adam/update_full_connection_1/weights/Variable/ApplyAdam?^train/Adam/update_full_connection_2/weights/Variable/ApplyAdam>^train/Adam/update_full_connection_2/biases/Variable/ApplyAdam*
T0*5
_class+
)'loc:@full_connection_1/weights/Variable*
_output_shapes
: 
Ć
train/Adam/Assign_1Assigntrain/beta2_powertrain/Adam/mul_1*
use_locking( *
T0*5
_class+
)'loc:@full_connection_1/weights/Variable*
validate_shape(*
_output_shapes
: 
ō

train/AdamNoOp<^train/Adam/update_hidden_layer_1/weights/Variable/ApplyAdam;^train/Adam/update_hidden_layer_1/biases/Variable/ApplyAdam<^train/Adam/update_hidden_layer_2/weights/Variable/ApplyAdam;^train/Adam/update_hidden_layer_2/biases/Variable/ApplyAdam?^train/Adam/update_full_connection_1/weights/Variable/ApplyAdam?^train/Adam/update_full_connection_2/weights/Variable/ApplyAdam>^train/Adam/update_full_connection_2/biases/Variable/ApplyAdam^train/Adam/Assign^train/Adam/Assign_1
O
	Softmax_1SoftmaxSoftmax*
T0*'
_output_shapes
:’’’’’’’’’

Ķ
Merge/MergeSummaryMergeSummaryhidden_layer_1/weights/weightshidden_layer_1/biases/biaseshidden_layer_2/weights/weightshidden_layer_2/biases/biases!full_connection_1/weights/weightsfull_connection_1/biases/biases!full_connection_2/weights/weightsfull_connection_2/biases/biasesloss*
N	*
_output_shapes
: 
°
initNoOp'^hidden_layer_1/weights/Variable/Assign&^hidden_layer_1/biases/Variable/Assign'^hidden_layer_2/weights/Variable/Assign&^hidden_layer_2/biases/Variable/Assign*^full_connection_1/weights/Variable/Assign)^full_connection_1/biases/Variable/Assign*^full_connection_2/weights/Variable/Assign)^full_connection_2/biases/Variable/Assign^train/beta1_power/Assign^train/beta2_power/Assign,^hidden_layer_1/weights/Variable/Adam/Assign.^hidden_layer_1/weights/Variable/Adam_1/Assign+^hidden_layer_1/biases/Variable/Adam/Assign-^hidden_layer_1/biases/Variable/Adam_1/Assign,^hidden_layer_2/weights/Variable/Adam/Assign.^hidden_layer_2/weights/Variable/Adam_1/Assign+^hidden_layer_2/biases/Variable/Adam/Assign-^hidden_layer_2/biases/Variable/Adam_1/Assign/^full_connection_1/weights/Variable/Adam/Assign1^full_connection_1/weights/Variable/Adam_1/Assign/^full_connection_2/weights/Variable/Adam/Assign1^full_connection_2/weights/Variable/Adam_1/Assign.^full_connection_2/biases/Variable/Adam/Assign0^full_connection_2/biases/Variable/Adam_1/Assign
P

save/ConstConst*
valueB Bmodel*
dtype0*
_output_shapes
: 

save/StringJoin/inputs_1Const*<
value3B1 B+_temp_90c0d9c9a5904c6b9da73416482d7184/part*
dtype0*
_output_shapes
: 
u
save/StringJoin
StringJoin
save/Constsave/StringJoin/inputs_1*
	separator *
N*
_output_shapes
: 
Q
save/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
\
save/ShardedFilename/shardConst*
value	B : *
dtype0*
_output_shapes
: 
}
save/ShardedFilenameShardedFilenamesave/StringJoinsave/ShardedFilename/shardsave/num_shards*
_output_shapes
: 
Ę
save/SaveV2/tensor_namesConst*ł
valueļBģB!full_connection_1/biases/VariableB"full_connection_1/weights/VariableB'full_connection_1/weights/Variable/AdamB)full_connection_1/weights/Variable/Adam_1B!full_connection_2/biases/VariableB&full_connection_2/biases/Variable/AdamB(full_connection_2/biases/Variable/Adam_1B"full_connection_2/weights/VariableB'full_connection_2/weights/Variable/AdamB)full_connection_2/weights/Variable/Adam_1Bhidden_layer_1/biases/VariableB#hidden_layer_1/biases/Variable/AdamB%hidden_layer_1/biases/Variable/Adam_1Bhidden_layer_1/weights/VariableB$hidden_layer_1/weights/Variable/AdamB&hidden_layer_1/weights/Variable/Adam_1Bhidden_layer_2/biases/VariableB#hidden_layer_2/biases/Variable/AdamB%hidden_layer_2/biases/Variable/Adam_1Bhidden_layer_2/weights/VariableB$hidden_layer_2/weights/Variable/AdamB&hidden_layer_2/weights/Variable/Adam_1Btrain/beta1_powerBtrain/beta2_power*
dtype0*
_output_shapes
:

save/SaveV2/shape_and_slicesConst*C
value:B8B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:
ļ
save/SaveV2SaveV2save/ShardedFilenamesave/SaveV2/tensor_namessave/SaveV2/shape_and_slices!full_connection_1/biases/Variable"full_connection_1/weights/Variable'full_connection_1/weights/Variable/Adam)full_connection_1/weights/Variable/Adam_1!full_connection_2/biases/Variable&full_connection_2/biases/Variable/Adam(full_connection_2/biases/Variable/Adam_1"full_connection_2/weights/Variable'full_connection_2/weights/Variable/Adam)full_connection_2/weights/Variable/Adam_1hidden_layer_1/biases/Variable#hidden_layer_1/biases/Variable/Adam%hidden_layer_1/biases/Variable/Adam_1hidden_layer_1/weights/Variable$hidden_layer_1/weights/Variable/Adam&hidden_layer_1/weights/Variable/Adam_1hidden_layer_2/biases/Variable#hidden_layer_2/biases/Variable/Adam%hidden_layer_2/biases/Variable/Adam_1hidden_layer_2/weights/Variable$hidden_layer_2/weights/Variable/Adam&hidden_layer_2/weights/Variable/Adam_1train/beta1_powertrain/beta2_power*&
dtypes
2

save/control_dependencyIdentitysave/ShardedFilename^save/SaveV2*
T0*'
_class
loc:@save/ShardedFilename*
_output_shapes
: 

+save/MergeV2Checkpoints/checkpoint_prefixesPacksave/ShardedFilename^save/control_dependency*
T0*

axis *
N*
_output_shapes
:
}
save/MergeV2CheckpointsMergeV2Checkpoints+save/MergeV2Checkpoints/checkpoint_prefixes
save/Const*
delete_old_dirs(
z
save/IdentityIdentity
save/Const^save/control_dependency^save/MergeV2Checkpoints*
T0*
_output_shapes
: 

save/RestoreV2/tensor_namesConst*6
value-B+B!full_connection_1/biases/Variable*
dtype0*
_output_shapes
:
h
save/RestoreV2/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices*
dtypes
2*
_output_shapes
:
Ķ
save/AssignAssign!full_connection_1/biases/Variablesave/RestoreV2*
use_locking(*
T0*4
_class*
(&loc:@full_connection_1/biases/Variable*
validate_shape(*
_output_shapes	
:

save/RestoreV2_1/tensor_namesConst*7
value.B,B"full_connection_1/weights/Variable*
dtype0*
_output_shapes
:
j
!save/RestoreV2_1/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save/RestoreV2_1	RestoreV2
save/Constsave/RestoreV2_1/tensor_names!save/RestoreV2_1/shape_and_slices*
dtypes
2*
_output_shapes
:
Ų
save/Assign_1Assign"full_connection_1/weights/Variablesave/RestoreV2_1*
use_locking(*
T0*5
_class+
)'loc:@full_connection_1/weights/Variable*
validate_shape(* 
_output_shapes
:
Ą

save/RestoreV2_2/tensor_namesConst*<
value3B1B'full_connection_1/weights/Variable/Adam*
dtype0*
_output_shapes
:
j
!save/RestoreV2_2/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save/RestoreV2_2	RestoreV2
save/Constsave/RestoreV2_2/tensor_names!save/RestoreV2_2/shape_and_slices*
dtypes
2*
_output_shapes
:
Ż
save/Assign_2Assign'full_connection_1/weights/Variable/Adamsave/RestoreV2_2*
use_locking(*
T0*5
_class+
)'loc:@full_connection_1/weights/Variable*
validate_shape(* 
_output_shapes
:
Ą

save/RestoreV2_3/tensor_namesConst*>
value5B3B)full_connection_1/weights/Variable/Adam_1*
dtype0*
_output_shapes
:
j
!save/RestoreV2_3/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save/RestoreV2_3	RestoreV2
save/Constsave/RestoreV2_3/tensor_names!save/RestoreV2_3/shape_and_slices*
dtypes
2*
_output_shapes
:
ß
save/Assign_3Assign)full_connection_1/weights/Variable/Adam_1save/RestoreV2_3*
use_locking(*
T0*5
_class+
)'loc:@full_connection_1/weights/Variable*
validate_shape(* 
_output_shapes
:
Ą

save/RestoreV2_4/tensor_namesConst*6
value-B+B!full_connection_2/biases/Variable*
dtype0*
_output_shapes
:
j
!save/RestoreV2_4/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save/RestoreV2_4	RestoreV2
save/Constsave/RestoreV2_4/tensor_names!save/RestoreV2_4/shape_and_slices*
dtypes
2*
_output_shapes
:
Š
save/Assign_4Assign!full_connection_2/biases/Variablesave/RestoreV2_4*
use_locking(*
T0*4
_class*
(&loc:@full_connection_2/biases/Variable*
validate_shape(*
_output_shapes
:


save/RestoreV2_5/tensor_namesConst*;
value2B0B&full_connection_2/biases/Variable/Adam*
dtype0*
_output_shapes
:
j
!save/RestoreV2_5/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save/RestoreV2_5	RestoreV2
save/Constsave/RestoreV2_5/tensor_names!save/RestoreV2_5/shape_and_slices*
dtypes
2*
_output_shapes
:
Õ
save/Assign_5Assign&full_connection_2/biases/Variable/Adamsave/RestoreV2_5*
use_locking(*
T0*4
_class*
(&loc:@full_connection_2/biases/Variable*
validate_shape(*
_output_shapes
:


save/RestoreV2_6/tensor_namesConst*=
value4B2B(full_connection_2/biases/Variable/Adam_1*
dtype0*
_output_shapes
:
j
!save/RestoreV2_6/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save/RestoreV2_6	RestoreV2
save/Constsave/RestoreV2_6/tensor_names!save/RestoreV2_6/shape_and_slices*
dtypes
2*
_output_shapes
:
×
save/Assign_6Assign(full_connection_2/biases/Variable/Adam_1save/RestoreV2_6*
use_locking(*
T0*4
_class*
(&loc:@full_connection_2/biases/Variable*
validate_shape(*
_output_shapes
:


save/RestoreV2_7/tensor_namesConst*7
value.B,B"full_connection_2/weights/Variable*
dtype0*
_output_shapes
:
j
!save/RestoreV2_7/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save/RestoreV2_7	RestoreV2
save/Constsave/RestoreV2_7/tensor_names!save/RestoreV2_7/shape_and_slices*
dtypes
2*
_output_shapes
:
×
save/Assign_7Assign"full_connection_2/weights/Variablesave/RestoreV2_7*
use_locking(*
T0*5
_class+
)'loc:@full_connection_2/weights/Variable*
validate_shape(*
_output_shapes
:	


save/RestoreV2_8/tensor_namesConst*<
value3B1B'full_connection_2/weights/Variable/Adam*
dtype0*
_output_shapes
:
j
!save/RestoreV2_8/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save/RestoreV2_8	RestoreV2
save/Constsave/RestoreV2_8/tensor_names!save/RestoreV2_8/shape_and_slices*
dtypes
2*
_output_shapes
:
Ü
save/Assign_8Assign'full_connection_2/weights/Variable/Adamsave/RestoreV2_8*
use_locking(*
T0*5
_class+
)'loc:@full_connection_2/weights/Variable*
validate_shape(*
_output_shapes
:	


save/RestoreV2_9/tensor_namesConst*>
value5B3B)full_connection_2/weights/Variable/Adam_1*
dtype0*
_output_shapes
:
j
!save/RestoreV2_9/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save/RestoreV2_9	RestoreV2
save/Constsave/RestoreV2_9/tensor_names!save/RestoreV2_9/shape_and_slices*
dtypes
2*
_output_shapes
:
Ž
save/Assign_9Assign)full_connection_2/weights/Variable/Adam_1save/RestoreV2_9*
use_locking(*
T0*5
_class+
)'loc:@full_connection_2/weights/Variable*
validate_shape(*
_output_shapes
:	


save/RestoreV2_10/tensor_namesConst*3
value*B(Bhidden_layer_1/biases/Variable*
dtype0*
_output_shapes
:
k
"save/RestoreV2_10/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save/RestoreV2_10	RestoreV2
save/Constsave/RestoreV2_10/tensor_names"save/RestoreV2_10/shape_and_slices*
dtypes
2*
_output_shapes
:
Ģ
save/Assign_10Assignhidden_layer_1/biases/Variablesave/RestoreV2_10*
use_locking(*
T0*1
_class'
%#loc:@hidden_layer_1/biases/Variable*
validate_shape(*
_output_shapes
: 

save/RestoreV2_11/tensor_namesConst*8
value/B-B#hidden_layer_1/biases/Variable/Adam*
dtype0*
_output_shapes
:
k
"save/RestoreV2_11/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save/RestoreV2_11	RestoreV2
save/Constsave/RestoreV2_11/tensor_names"save/RestoreV2_11/shape_and_slices*
dtypes
2*
_output_shapes
:
Ń
save/Assign_11Assign#hidden_layer_1/biases/Variable/Adamsave/RestoreV2_11*
use_locking(*
T0*1
_class'
%#loc:@hidden_layer_1/biases/Variable*
validate_shape(*
_output_shapes
: 

save/RestoreV2_12/tensor_namesConst*:
value1B/B%hidden_layer_1/biases/Variable/Adam_1*
dtype0*
_output_shapes
:
k
"save/RestoreV2_12/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save/RestoreV2_12	RestoreV2
save/Constsave/RestoreV2_12/tensor_names"save/RestoreV2_12/shape_and_slices*
dtypes
2*
_output_shapes
:
Ó
save/Assign_12Assign%hidden_layer_1/biases/Variable/Adam_1save/RestoreV2_12*
use_locking(*
T0*1
_class'
%#loc:@hidden_layer_1/biases/Variable*
validate_shape(*
_output_shapes
: 

save/RestoreV2_13/tensor_namesConst*4
value+B)Bhidden_layer_1/weights/Variable*
dtype0*
_output_shapes
:
k
"save/RestoreV2_13/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save/RestoreV2_13	RestoreV2
save/Constsave/RestoreV2_13/tensor_names"save/RestoreV2_13/shape_and_slices*
dtypes
2*
_output_shapes
:
Ś
save/Assign_13Assignhidden_layer_1/weights/Variablesave/RestoreV2_13*
use_locking(*
T0*2
_class(
&$loc:@hidden_layer_1/weights/Variable*
validate_shape(*&
_output_shapes
: 

save/RestoreV2_14/tensor_namesConst*9
value0B.B$hidden_layer_1/weights/Variable/Adam*
dtype0*
_output_shapes
:
k
"save/RestoreV2_14/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save/RestoreV2_14	RestoreV2
save/Constsave/RestoreV2_14/tensor_names"save/RestoreV2_14/shape_and_slices*
dtypes
2*
_output_shapes
:
ß
save/Assign_14Assign$hidden_layer_1/weights/Variable/Adamsave/RestoreV2_14*
use_locking(*
T0*2
_class(
&$loc:@hidden_layer_1/weights/Variable*
validate_shape(*&
_output_shapes
: 

save/RestoreV2_15/tensor_namesConst*;
value2B0B&hidden_layer_1/weights/Variable/Adam_1*
dtype0*
_output_shapes
:
k
"save/RestoreV2_15/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save/RestoreV2_15	RestoreV2
save/Constsave/RestoreV2_15/tensor_names"save/RestoreV2_15/shape_and_slices*
dtypes
2*
_output_shapes
:
į
save/Assign_15Assign&hidden_layer_1/weights/Variable/Adam_1save/RestoreV2_15*
use_locking(*
T0*2
_class(
&$loc:@hidden_layer_1/weights/Variable*
validate_shape(*&
_output_shapes
: 

save/RestoreV2_16/tensor_namesConst*3
value*B(Bhidden_layer_2/biases/Variable*
dtype0*
_output_shapes
:
k
"save/RestoreV2_16/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save/RestoreV2_16	RestoreV2
save/Constsave/RestoreV2_16/tensor_names"save/RestoreV2_16/shape_and_slices*
dtypes
2*
_output_shapes
:
Ģ
save/Assign_16Assignhidden_layer_2/biases/Variablesave/RestoreV2_16*
use_locking(*
T0*1
_class'
%#loc:@hidden_layer_2/biases/Variable*
validate_shape(*
_output_shapes
:@

save/RestoreV2_17/tensor_namesConst*8
value/B-B#hidden_layer_2/biases/Variable/Adam*
dtype0*
_output_shapes
:
k
"save/RestoreV2_17/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save/RestoreV2_17	RestoreV2
save/Constsave/RestoreV2_17/tensor_names"save/RestoreV2_17/shape_and_slices*
dtypes
2*
_output_shapes
:
Ń
save/Assign_17Assign#hidden_layer_2/biases/Variable/Adamsave/RestoreV2_17*
use_locking(*
T0*1
_class'
%#loc:@hidden_layer_2/biases/Variable*
validate_shape(*
_output_shapes
:@

save/RestoreV2_18/tensor_namesConst*:
value1B/B%hidden_layer_2/biases/Variable/Adam_1*
dtype0*
_output_shapes
:
k
"save/RestoreV2_18/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save/RestoreV2_18	RestoreV2
save/Constsave/RestoreV2_18/tensor_names"save/RestoreV2_18/shape_and_slices*
dtypes
2*
_output_shapes
:
Ó
save/Assign_18Assign%hidden_layer_2/biases/Variable/Adam_1save/RestoreV2_18*
use_locking(*
T0*1
_class'
%#loc:@hidden_layer_2/biases/Variable*
validate_shape(*
_output_shapes
:@

save/RestoreV2_19/tensor_namesConst*4
value+B)Bhidden_layer_2/weights/Variable*
dtype0*
_output_shapes
:
k
"save/RestoreV2_19/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save/RestoreV2_19	RestoreV2
save/Constsave/RestoreV2_19/tensor_names"save/RestoreV2_19/shape_and_slices*
dtypes
2*
_output_shapes
:
Ś
save/Assign_19Assignhidden_layer_2/weights/Variablesave/RestoreV2_19*
use_locking(*
T0*2
_class(
&$loc:@hidden_layer_2/weights/Variable*
validate_shape(*&
_output_shapes
: @

save/RestoreV2_20/tensor_namesConst*9
value0B.B$hidden_layer_2/weights/Variable/Adam*
dtype0*
_output_shapes
:
k
"save/RestoreV2_20/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save/RestoreV2_20	RestoreV2
save/Constsave/RestoreV2_20/tensor_names"save/RestoreV2_20/shape_and_slices*
dtypes
2*
_output_shapes
:
ß
save/Assign_20Assign$hidden_layer_2/weights/Variable/Adamsave/RestoreV2_20*
use_locking(*
T0*2
_class(
&$loc:@hidden_layer_2/weights/Variable*
validate_shape(*&
_output_shapes
: @

save/RestoreV2_21/tensor_namesConst*;
value2B0B&hidden_layer_2/weights/Variable/Adam_1*
dtype0*
_output_shapes
:
k
"save/RestoreV2_21/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save/RestoreV2_21	RestoreV2
save/Constsave/RestoreV2_21/tensor_names"save/RestoreV2_21/shape_and_slices*
dtypes
2*
_output_shapes
:
į
save/Assign_21Assign&hidden_layer_2/weights/Variable/Adam_1save/RestoreV2_21*
use_locking(*
T0*2
_class(
&$loc:@hidden_layer_2/weights/Variable*
validate_shape(*&
_output_shapes
: @
x
save/RestoreV2_22/tensor_namesConst*&
valueBBtrain/beta1_power*
dtype0*
_output_shapes
:
k
"save/RestoreV2_22/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save/RestoreV2_22	RestoreV2
save/Constsave/RestoreV2_22/tensor_names"save/RestoreV2_22/shape_and_slices*
dtypes
2*
_output_shapes
:
æ
save/Assign_22Assigntrain/beta1_powersave/RestoreV2_22*
use_locking(*
T0*5
_class+
)'loc:@full_connection_1/weights/Variable*
validate_shape(*
_output_shapes
: 
x
save/RestoreV2_23/tensor_namesConst*&
valueBBtrain/beta2_power*
dtype0*
_output_shapes
:
k
"save/RestoreV2_23/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save/RestoreV2_23	RestoreV2
save/Constsave/RestoreV2_23/tensor_names"save/RestoreV2_23/shape_and_slices*
dtypes
2*
_output_shapes
:
æ
save/Assign_23Assigntrain/beta2_powersave/RestoreV2_23*
use_locking(*
T0*5
_class+
)'loc:@full_connection_1/weights/Variable*
validate_shape(*
_output_shapes
: 
¦
save/restore_shardNoOp^save/Assign^save/Assign_1^save/Assign_2^save/Assign_3^save/Assign_4^save/Assign_5^save/Assign_6^save/Assign_7^save/Assign_8^save/Assign_9^save/Assign_10^save/Assign_11^save/Assign_12^save/Assign_13^save/Assign_14^save/Assign_15^save/Assign_16^save/Assign_17^save/Assign_18^save/Assign_19^save/Assign_20^save/Assign_21^save/Assign_22^save/Assign_23
-
save/restore_allNoOp^save/restore_shard "<
save/Const:0save/Identity:0save/restore_all (5 @F8"

trainable_variables’	ü	

!hidden_layer_1/weights/Variable:0&hidden_layer_1/weights/Variable/Assign&hidden_layer_1/weights/Variable/read:02)hidden_layer_1/weights/truncated_normal:0

 hidden_layer_1/biases/Variable:0%hidden_layer_1/biases/Variable/Assign%hidden_layer_1/biases/Variable/read:02hidden_layer_1/biases/Const:0

!hidden_layer_2/weights/Variable:0&hidden_layer_2/weights/Variable/Assign&hidden_layer_2/weights/Variable/read:02)hidden_layer_2/weights/truncated_normal:0

 hidden_layer_2/biases/Variable:0%hidden_layer_2/biases/Variable/Assign%hidden_layer_2/biases/Variable/read:02hidden_layer_2/biases/Const:0
Ŗ
$full_connection_1/weights/Variable:0)full_connection_1/weights/Variable/Assign)full_connection_1/weights/Variable/read:02,full_connection_1/weights/truncated_normal:0

#full_connection_1/biases/Variable:0(full_connection_1/biases/Variable/Assign(full_connection_1/biases/Variable/read:02 full_connection_1/biases/Const:0
Ŗ
$full_connection_2/weights/Variable:0)full_connection_2/weights/Variable/Assign)full_connection_2/weights/Variable/read:02,full_connection_2/weights/truncated_normal:0

#full_connection_2/biases/Variable:0(full_connection_2/biases/Variable/Assign(full_connection_2/biases/Variable/read:02 full_connection_2/biases/Const:0"­
	summaries

 hidden_layer_1/weights/weights:0
hidden_layer_1/biases/biases:0
 hidden_layer_2/weights/weights:0
hidden_layer_2/biases/biases:0
#full_connection_1/weights/weights:0
!full_connection_1/biases/biases:0
#full_connection_2/weights/weights:0
!full_connection_2/biases/biases:0
loss:0"
train_op


train/Adam"Ć!
	variablesµ!²!

!hidden_layer_1/weights/Variable:0&hidden_layer_1/weights/Variable/Assign&hidden_layer_1/weights/Variable/read:02)hidden_layer_1/weights/truncated_normal:0

 hidden_layer_1/biases/Variable:0%hidden_layer_1/biases/Variable/Assign%hidden_layer_1/biases/Variable/read:02hidden_layer_1/biases/Const:0

!hidden_layer_2/weights/Variable:0&hidden_layer_2/weights/Variable/Assign&hidden_layer_2/weights/Variable/read:02)hidden_layer_2/weights/truncated_normal:0

 hidden_layer_2/biases/Variable:0%hidden_layer_2/biases/Variable/Assign%hidden_layer_2/biases/Variable/read:02hidden_layer_2/biases/Const:0
Ŗ
$full_connection_1/weights/Variable:0)full_connection_1/weights/Variable/Assign)full_connection_1/weights/Variable/read:02,full_connection_1/weights/truncated_normal:0

#full_connection_1/biases/Variable:0(full_connection_1/biases/Variable/Assign(full_connection_1/biases/Variable/read:02 full_connection_1/biases/Const:0
Ŗ
$full_connection_2/weights/Variable:0)full_connection_2/weights/Variable/Assign)full_connection_2/weights/Variable/read:02,full_connection_2/weights/truncated_normal:0

#full_connection_2/biases/Variable:0(full_connection_2/biases/Variable/Assign(full_connection_2/biases/Variable/read:02 full_connection_2/biases/Const:0
l
train/beta1_power:0train/beta1_power/Assigntrain/beta1_power/read:02!train/beta1_power/initial_value:0
l
train/beta2_power:0train/beta2_power/Assigntrain/beta2_power/read:02!train/beta2_power/initial_value:0
¼
&hidden_layer_1/weights/Variable/Adam:0+hidden_layer_1/weights/Variable/Adam/Assign+hidden_layer_1/weights/Variable/Adam/read:028hidden_layer_1/weights/Variable/Adam/Initializer/zeros:0
Ä
(hidden_layer_1/weights/Variable/Adam_1:0-hidden_layer_1/weights/Variable/Adam_1/Assign-hidden_layer_1/weights/Variable/Adam_1/read:02:hidden_layer_1/weights/Variable/Adam_1/Initializer/zeros:0
ø
%hidden_layer_1/biases/Variable/Adam:0*hidden_layer_1/biases/Variable/Adam/Assign*hidden_layer_1/biases/Variable/Adam/read:027hidden_layer_1/biases/Variable/Adam/Initializer/zeros:0
Ą
'hidden_layer_1/biases/Variable/Adam_1:0,hidden_layer_1/biases/Variable/Adam_1/Assign,hidden_layer_1/biases/Variable/Adam_1/read:029hidden_layer_1/biases/Variable/Adam_1/Initializer/zeros:0
¼
&hidden_layer_2/weights/Variable/Adam:0+hidden_layer_2/weights/Variable/Adam/Assign+hidden_layer_2/weights/Variable/Adam/read:028hidden_layer_2/weights/Variable/Adam/Initializer/zeros:0
Ä
(hidden_layer_2/weights/Variable/Adam_1:0-hidden_layer_2/weights/Variable/Adam_1/Assign-hidden_layer_2/weights/Variable/Adam_1/read:02:hidden_layer_2/weights/Variable/Adam_1/Initializer/zeros:0
ø
%hidden_layer_2/biases/Variable/Adam:0*hidden_layer_2/biases/Variable/Adam/Assign*hidden_layer_2/biases/Variable/Adam/read:027hidden_layer_2/biases/Variable/Adam/Initializer/zeros:0
Ą
'hidden_layer_2/biases/Variable/Adam_1:0,hidden_layer_2/biases/Variable/Adam_1/Assign,hidden_layer_2/biases/Variable/Adam_1/read:029hidden_layer_2/biases/Variable/Adam_1/Initializer/zeros:0
Č
)full_connection_1/weights/Variable/Adam:0.full_connection_1/weights/Variable/Adam/Assign.full_connection_1/weights/Variable/Adam/read:02;full_connection_1/weights/Variable/Adam/Initializer/zeros:0
Š
+full_connection_1/weights/Variable/Adam_1:00full_connection_1/weights/Variable/Adam_1/Assign0full_connection_1/weights/Variable/Adam_1/read:02=full_connection_1/weights/Variable/Adam_1/Initializer/zeros:0
Č
)full_connection_2/weights/Variable/Adam:0.full_connection_2/weights/Variable/Adam/Assign.full_connection_2/weights/Variable/Adam/read:02;full_connection_2/weights/Variable/Adam/Initializer/zeros:0
Š
+full_connection_2/weights/Variable/Adam_1:00full_connection_2/weights/Variable/Adam_1/Assign0full_connection_2/weights/Variable/Adam_1/read:02=full_connection_2/weights/Variable/Adam_1/Initializer/zeros:0
Ä
(full_connection_2/biases/Variable/Adam:0-full_connection_2/biases/Variable/Adam/Assign-full_connection_2/biases/Variable/Adam/read:02:full_connection_2/biases/Variable/Adam/Initializer/zeros:0
Ģ
*full_connection_2/biases/Variable/Adam_1:0/full_connection_2/biases/Variable/Adam_1/Assign/full_connection_2/biases/Variable/Adam_1/read:02<full_connection_2/biases/Variable/Adam_1/Initializer/zeros:0*
Infer
6
inputs,
image_input:0’’’’’’’’’*
outputs
predict_op:0	’’’’’’’’’tensorflow/serving/predict