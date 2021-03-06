??
??
B
AddV2
x"T
y"T
z"T"
Ttype:
2	??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
8
Const
output"dtype"
valuetensor"
dtypetype
?
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

?
FusedBatchNormV3
x"T

scale"U
offset"U	
mean"U
variance"U
y"T

batch_mean"U
batch_variance"U
reserve_space_1"U
reserve_space_2"U
reserve_space_3"U"
Ttype:
2"
Utype:
2"
epsilonfloat%??8"&
exponential_avg_factorfloat%  ??";
data_formatstringNHWC:
NHWCNCHWNDHWCNCDHW"
is_trainingbool(
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
?
MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
:
Maximum
x"T
y"T
z"T"
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?
:
Minimum
x"T
y"T
z"T"
Ttype:

2	
=
Mul
x"T
y"T
z"T"
Ttype:
2	?
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
@
ReadVariableOp
resource
value"dtype"
dtypetype?
>
RealDiv
x"T
y"T
z"T"
Ttype:
2	
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
list(type)(0?
2
Round
x"T
y"T"
Ttype:
2
	
.
Rsqrt
x"T
y"T"
Ttype:

2
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ?
@
StaticRegexFullMatch	
input

output
"
patternstring
2
StopGradient

input"T
output"T"	
Ttype
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
;
Sub
x"T
y"T
z"T"
Ttype:
2	
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.4.12v2.4.0-49-g85c8b2a817f8??
|
conv1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv1/kernel
u
 conv1/kernel/Read/ReadVariableOpReadVariableOpconv1/kernel*&
_output_shapes
: *
dtype0
j
	bn1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	bn1/gamma
c
bn1/gamma/Read/ReadVariableOpReadVariableOp	bn1/gamma*
_output_shapes
: *
dtype0
h
bn1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
bn1/beta
a
bn1/beta/Read/ReadVariableOpReadVariableOpbn1/beta*
_output_shapes
: *
dtype0
v
bn1/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: * 
shared_namebn1/moving_mean
o
#bn1/moving_mean/Read/ReadVariableOpReadVariableOpbn1/moving_mean*
_output_shapes
: *
dtype0
~
bn1/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *$
shared_namebn1/moving_variance
w
'bn1/moving_variance/Read/ReadVariableOpReadVariableOpbn1/moving_variance*
_output_shapes
: *
dtype0
?
binary_conv2d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *%
shared_namebinary_conv2d/kernel
?
(binary_conv2d/kernel/Read/ReadVariableOpReadVariableOpbinary_conv2d/kernel*&
_output_shapes
:  *
dtype0
?
batch_normalization/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_namebatch_normalization/gamma
?
-batch_normalization/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization/gamma*
_output_shapes
:*
dtype0
?
batch_normalization/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_namebatch_normalization/beta
?
,batch_normalization/beta/Read/ReadVariableOpReadVariableOpbatch_normalization/beta*
_output_shapes
:*
dtype0
?
batch_normalization/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!batch_normalization/moving_mean
?
3batch_normalization/moving_mean/Read/ReadVariableOpReadVariableOpbatch_normalization/moving_mean*
_output_shapes
:*
dtype0
?
#batch_normalization/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#batch_normalization/moving_variance
?
7batch_normalization/moving_variance/Read/ReadVariableOpReadVariableOp#batch_normalization/moving_variance*
_output_shapes
:*
dtype0
x
dense5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
?@?*
shared_namedense5/kernel
q
!dense5/kernel/Read/ReadVariableOpReadVariableOpdense5/kernel* 
_output_shapes
:
?@?*
dtype0
k
	bn5/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_name	bn5/gamma
d
bn5/gamma/Read/ReadVariableOpReadVariableOp	bn5/gamma*
_output_shapes	
:?*
dtype0
i
bn5/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_name
bn5/beta
b
bn5/beta/Read/ReadVariableOpReadVariableOpbn5/beta*
_output_shapes	
:?*
dtype0
w
bn5/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:?* 
shared_namebn5/moving_mean
p
#bn5/moving_mean/Read/ReadVariableOpReadVariableOpbn5/moving_mean*
_output_shapes	
:?*
dtype0

bn5/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*$
shared_namebn5/moving_variance
x
'bn5/moving_variance/Read/ReadVariableOpReadVariableOpbn5/moving_variance*
_output_shapes	
:?*
dtype0
w
dense6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?d*
shared_namedense6/kernel
p
!dense6/kernel/Read/ReadVariableOpReadVariableOpdense6/kernel*
_output_shapes
:	?d*
dtype0
j
	bn6/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*
shared_name	bn6/gamma
c
bn6/gamma/Read/ReadVariableOpReadVariableOp	bn6/gamma*
_output_shapes
:d*
dtype0
h
bn6/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*
shared_name
bn6/beta
a
bn6/beta/Read/ReadVariableOpReadVariableOpbn6/beta*
_output_shapes
:d*
dtype0
v
bn6/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:d* 
shared_namebn6/moving_mean
o
#bn6/moving_mean/Read/ReadVariableOpReadVariableOpbn6/moving_mean*
_output_shapes
:d*
dtype0
~
bn6/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*$
shared_namebn6/moving_variance
w
'bn6/moving_variance/Read/ReadVariableOpReadVariableOpbn6/moving_variance*
_output_shapes
:d*
dtype0
`
beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namebeta_1
Y
beta_1/Read/ReadVariableOpReadVariableOpbeta_1*
_output_shapes
: *
dtype0
`
beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namebeta_2
Y
beta_2/Read/ReadVariableOpReadVariableOpbeta_2*
_output_shapes
: *
dtype0
^
decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedecay
W
decay/Read/ReadVariableOpReadVariableOpdecay*
_output_shapes
: *
dtype0
n
learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namelearning_rate
g
!learning_rate/Read/ReadVariableOpReadVariableOplearning_rate*
_output_shapes
: *
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
?
Adam/conv1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *$
shared_nameAdam/conv1/kernel/m
?
'Adam/conv1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv1/kernel/m*&
_output_shapes
: *
dtype0
x
Adam/bn1/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *!
shared_nameAdam/bn1/gamma/m
q
$Adam/bn1/gamma/m/Read/ReadVariableOpReadVariableOpAdam/bn1/gamma/m*
_output_shapes
: *
dtype0
v
Adam/bn1/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: * 
shared_nameAdam/bn1/beta/m
o
#Adam/bn1/beta/m/Read/ReadVariableOpReadVariableOpAdam/bn1/beta/m*
_output_shapes
: *
dtype0
?
Adam/binary_conv2d/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *,
shared_nameAdam/binary_conv2d/kernel/m
?
/Adam/binary_conv2d/kernel/m/Read/ReadVariableOpReadVariableOpAdam/binary_conv2d/kernel/m*&
_output_shapes
:  *
dtype0
?
 Adam/batch_normalization/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" Adam/batch_normalization/gamma/m
?
4Adam/batch_normalization/gamma/m/Read/ReadVariableOpReadVariableOp Adam/batch_normalization/gamma/m*
_output_shapes
:*
dtype0
?
Adam/batch_normalization/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!Adam/batch_normalization/beta/m
?
3Adam/batch_normalization/beta/m/Read/ReadVariableOpReadVariableOpAdam/batch_normalization/beta/m*
_output_shapes
:*
dtype0
?
Adam/dense5/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
?@?*%
shared_nameAdam/dense5/kernel/m

(Adam/dense5/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense5/kernel/m* 
_output_shapes
:
?@?*
dtype0
y
Adam/bn5/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*!
shared_nameAdam/bn5/gamma/m
r
$Adam/bn5/gamma/m/Read/ReadVariableOpReadVariableOpAdam/bn5/gamma/m*
_output_shapes	
:?*
dtype0
w
Adam/bn5/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?* 
shared_nameAdam/bn5/beta/m
p
#Adam/bn5/beta/m/Read/ReadVariableOpReadVariableOpAdam/bn5/beta/m*
_output_shapes	
:?*
dtype0
?
Adam/dense6/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?d*%
shared_nameAdam/dense6/kernel/m
~
(Adam/dense6/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense6/kernel/m*
_output_shapes
:	?d*
dtype0
x
Adam/bn6/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*!
shared_nameAdam/bn6/gamma/m
q
$Adam/bn6/gamma/m/Read/ReadVariableOpReadVariableOpAdam/bn6/gamma/m*
_output_shapes
:d*
dtype0
v
Adam/bn6/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:d* 
shared_nameAdam/bn6/beta/m
o
#Adam/bn6/beta/m/Read/ReadVariableOpReadVariableOpAdam/bn6/beta/m*
_output_shapes
:d*
dtype0
?
Adam/conv1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *$
shared_nameAdam/conv1/kernel/v
?
'Adam/conv1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv1/kernel/v*&
_output_shapes
: *
dtype0
x
Adam/bn1/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *!
shared_nameAdam/bn1/gamma/v
q
$Adam/bn1/gamma/v/Read/ReadVariableOpReadVariableOpAdam/bn1/gamma/v*
_output_shapes
: *
dtype0
v
Adam/bn1/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: * 
shared_nameAdam/bn1/beta/v
o
#Adam/bn1/beta/v/Read/ReadVariableOpReadVariableOpAdam/bn1/beta/v*
_output_shapes
: *
dtype0
?
Adam/binary_conv2d/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *,
shared_nameAdam/binary_conv2d/kernel/v
?
/Adam/binary_conv2d/kernel/v/Read/ReadVariableOpReadVariableOpAdam/binary_conv2d/kernel/v*&
_output_shapes
:  *
dtype0
?
 Adam/batch_normalization/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" Adam/batch_normalization/gamma/v
?
4Adam/batch_normalization/gamma/v/Read/ReadVariableOpReadVariableOp Adam/batch_normalization/gamma/v*
_output_shapes
:*
dtype0
?
Adam/batch_normalization/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!Adam/batch_normalization/beta/v
?
3Adam/batch_normalization/beta/v/Read/ReadVariableOpReadVariableOpAdam/batch_normalization/beta/v*
_output_shapes
:*
dtype0
?
Adam/dense5/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
?@?*%
shared_nameAdam/dense5/kernel/v

(Adam/dense5/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense5/kernel/v* 
_output_shapes
:
?@?*
dtype0
y
Adam/bn5/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*!
shared_nameAdam/bn5/gamma/v
r
$Adam/bn5/gamma/v/Read/ReadVariableOpReadVariableOpAdam/bn5/gamma/v*
_output_shapes	
:?*
dtype0
w
Adam/bn5/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?* 
shared_nameAdam/bn5/beta/v
p
#Adam/bn5/beta/v/Read/ReadVariableOpReadVariableOpAdam/bn5/beta/v*
_output_shapes	
:?*
dtype0
?
Adam/dense6/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?d*%
shared_nameAdam/dense6/kernel/v
~
(Adam/dense6/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense6/kernel/v*
_output_shapes
:	?d*
dtype0
x
Adam/bn6/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*!
shared_nameAdam/bn6/gamma/v
q
$Adam/bn6/gamma/v/Read/ReadVariableOpReadVariableOpAdam/bn6/gamma/v*
_output_shapes
:d*
dtype0
v
Adam/bn6/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:d* 
shared_nameAdam/bn6/beta/v
o
#Adam/bn6/beta/v/Read/ReadVariableOpReadVariableOpAdam/bn6/beta/v*
_output_shapes
:d*
dtype0

NoOpNoOp
?]
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?]
value?]B?] B?]
?
layer_with_weights-0
layer-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer-5
layer_with_weights-3
layer-6
layer-7
	layer-8

layer_with_weights-4

layer-9
layer_with_weights-5
layer-10
layer-11
layer_with_weights-6
layer-12
layer_with_weights-7
layer-13
	optimizer
regularization_losses
trainable_variables
	variables
	keras_api

signatures
r

kernel
lr_multipliers
regularization_losses
trainable_variables
	variables
	keras_api
R
regularization_losses
trainable_variables
	variables
	keras_api
?
axis
	 gamma
!beta
"moving_mean
#moving_variance
$regularization_losses
%trainable_variables
&	variables
'	keras_api
R
(regularization_losses
)trainable_variables
*	variables
+	keras_api
r

,kernel
-lr_multipliers
.regularization_losses
/trainable_variables
0	variables
1	keras_api
R
2regularization_losses
3trainable_variables
4	variables
5	keras_api
?
6axis
	7gamma
8beta
9moving_mean
:moving_variance
;regularization_losses
<trainable_variables
=	variables
>	keras_api
R
?regularization_losses
@trainable_variables
A	variables
B	keras_api
R
Cregularization_losses
Dtrainable_variables
E	variables
F	keras_api
r

Gkernel
Hlr_multipliers
Iregularization_losses
Jtrainable_variables
K	variables
L	keras_api
?
Maxis
	Ngamma
Obeta
Pmoving_mean
Qmoving_variance
Rregularization_losses
Strainable_variables
T	variables
U	keras_api
R
Vregularization_losses
Wtrainable_variables
X	variables
Y	keras_api
r

Zkernel
[lr_multipliers
\regularization_losses
]trainable_variables
^	variables
_	keras_api
?
`axis
	agamma
bbeta
cmoving_mean
dmoving_variance
eregularization_losses
ftrainable_variables
g	variables
h	keras_api
?

ibeta_1

jbeta_2
	kdecay
llearning_rate
miterm? m?!m?,m?7m?8m?Gm?Nm?Om?Zm?am?bm?v? v?!v?,v?7v?8v?Gv?Nv?Ov?Zv?av?bv?
 
V
0
 1
!2
,3
74
85
G6
N7
O8
Z9
a10
b11
?
0
 1
!2
"3
#4
,5
76
87
98
:9
G10
N11
O12
P13
Q14
Z15
a16
b17
c18
d19
?
nnon_trainable_variables
regularization_losses
olayer_regularization_losses

players
qmetrics
rlayer_metrics
trainable_variables
	variables
 
XV
VARIABLE_VALUEconv1/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
 
 

0

0
?
snon_trainable_variables
regularization_losses
tlayer_regularization_losses

ulayers
vmetrics
wlayer_metrics
trainable_variables
	variables
 
 
 
?
xnon_trainable_variables
regularization_losses
ylayer_regularization_losses

zlayers
{metrics
|layer_metrics
trainable_variables
	variables
 
TR
VARIABLE_VALUE	bn1/gamma5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUE
RP
VARIABLE_VALUEbn1/beta4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUE
`^
VARIABLE_VALUEbn1/moving_mean;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
hf
VARIABLE_VALUEbn1/moving_variance?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

 0
!1

 0
!1
"2
#3
?
}non_trainable_variables
$regularization_losses
~layer_regularization_losses

layers
?metrics
?layer_metrics
%trainable_variables
&	variables
 
 
 
?
?non_trainable_variables
(regularization_losses
 ?layer_regularization_losses
?layers
?metrics
?layer_metrics
)trainable_variables
*	variables
`^
VARIABLE_VALUEbinary_conv2d/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
 
 

,0

,0
?
?non_trainable_variables
.regularization_losses
 ?layer_regularization_losses
?layers
?metrics
?layer_metrics
/trainable_variables
0	variables
 
 
 
?
?non_trainable_variables
2regularization_losses
 ?layer_regularization_losses
?layers
?metrics
?layer_metrics
3trainable_variables
4	variables
 
db
VARIABLE_VALUEbatch_normalization/gamma5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUEbatch_normalization/beta4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEbatch_normalization/moving_mean;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUE#batch_normalization/moving_variance?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

70
81

70
81
92
:3
?
?non_trainable_variables
;regularization_losses
 ?layer_regularization_losses
?layers
?metrics
?layer_metrics
<trainable_variables
=	variables
 
 
 
?
?non_trainable_variables
?regularization_losses
 ?layer_regularization_losses
?layers
?metrics
?layer_metrics
@trainable_variables
A	variables
 
 
 
?
?non_trainable_variables
Cregularization_losses
 ?layer_regularization_losses
?layers
?metrics
?layer_metrics
Dtrainable_variables
E	variables
YW
VARIABLE_VALUEdense5/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
 
 

G0

G0
?
?non_trainable_variables
Iregularization_losses
 ?layer_regularization_losses
?layers
?metrics
?layer_metrics
Jtrainable_variables
K	variables
 
TR
VARIABLE_VALUE	bn5/gamma5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUE
RP
VARIABLE_VALUEbn5/beta4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUE
`^
VARIABLE_VALUEbn5/moving_mean;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
hf
VARIABLE_VALUEbn5/moving_variance?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

N0
O1

N0
O1
P2
Q3
?
?non_trainable_variables
Rregularization_losses
 ?layer_regularization_losses
?layers
?metrics
?layer_metrics
Strainable_variables
T	variables
 
 
 
?
?non_trainable_variables
Vregularization_losses
 ?layer_regularization_losses
?layers
?metrics
?layer_metrics
Wtrainable_variables
X	variables
YW
VARIABLE_VALUEdense6/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
 
 

Z0

Z0
?
?non_trainable_variables
\regularization_losses
 ?layer_regularization_losses
?layers
?metrics
?layer_metrics
]trainable_variables
^	variables
 
TR
VARIABLE_VALUE	bn6/gamma5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUE
RP
VARIABLE_VALUEbn6/beta4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUE
`^
VARIABLE_VALUEbn6/moving_mean;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
hf
VARIABLE_VALUEbn6/moving_variance?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

a0
b1

a0
b1
c2
d3
?
?non_trainable_variables
eregularization_losses
 ?layer_regularization_losses
?layers
?metrics
?layer_metrics
ftrainable_variables
g	variables
GE
VARIABLE_VALUEbeta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
GE
VARIABLE_VALUEbeta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
EC
VARIABLE_VALUEdecay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUElearning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
8
"0
#1
92
:3
P4
Q5
c6
d7
 
f
0
1
2
3
4
5
6
7
	8

9
10
11
12
13

?0
?1
 
 
 
 
 
 
 
 
 
 
 

"0
#1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

90
:1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

P0
Q1
 
 
 
 
 
 
 
 
 
 
 
 
 
 

c0
d1
 
 
 
 
8

?total

?count
?	variables
?	keras_api
I

?total

?count
?
_fn_kwargs
?	variables
?	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?	variables
{y
VARIABLE_VALUEAdam/conv1/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/bn1/gamma/mQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUEAdam/bn1/beta/mPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/binary_conv2d/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE Adam/batch_normalization/gamma/mQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/batch_normalization/beta/mPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/dense5/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/bn5/gamma/mQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUEAdam/bn5/beta/mPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/dense6/kernel/mRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/bn6/gamma/mQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUEAdam/bn6/beta/mPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv1/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/bn1/gamma/vQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUEAdam/bn1/beta/vPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/binary_conv2d/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE Adam/batch_normalization/gamma/vQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/batch_normalization/beta/vPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/dense5/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/bn5/gamma/vQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUEAdam/bn5/beta/vPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/dense6/kernel/vRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/bn6/gamma/vQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUEAdam/bn6/beta/vPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
serving_default_conv1_inputPlaceholder*/
_output_shapes
:?????????@@*
dtype0*$
shape:?????????@@
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_conv1_inputconv1/kernel	bn1/gammabn1/betabn1/moving_meanbn1/moving_variancebinary_conv2d/kernelbatch_normalization/gammabatch_normalization/betabatch_normalization/moving_mean#batch_normalization/moving_variancedense5/kernelbn5/moving_variance	bn5/gammabn5/moving_meanbn5/betadense6/kernelbn6/moving_variance	bn6/gammabn6/moving_meanbn6/beta* 
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d*6
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *,
f'R%
#__inference_signature_wrapper_26818
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename conv1/kernel/Read/ReadVariableOpbn1/gamma/Read/ReadVariableOpbn1/beta/Read/ReadVariableOp#bn1/moving_mean/Read/ReadVariableOp'bn1/moving_variance/Read/ReadVariableOp(binary_conv2d/kernel/Read/ReadVariableOp-batch_normalization/gamma/Read/ReadVariableOp,batch_normalization/beta/Read/ReadVariableOp3batch_normalization/moving_mean/Read/ReadVariableOp7batch_normalization/moving_variance/Read/ReadVariableOp!dense5/kernel/Read/ReadVariableOpbn5/gamma/Read/ReadVariableOpbn5/beta/Read/ReadVariableOp#bn5/moving_mean/Read/ReadVariableOp'bn5/moving_variance/Read/ReadVariableOp!dense6/kernel/Read/ReadVariableOpbn6/gamma/Read/ReadVariableOpbn6/beta/Read/ReadVariableOp#bn6/moving_mean/Read/ReadVariableOp'bn6/moving_variance/Read/ReadVariableOpbeta_1/Read/ReadVariableOpbeta_2/Read/ReadVariableOpdecay/Read/ReadVariableOp!learning_rate/Read/ReadVariableOpAdam/iter/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp'Adam/conv1/kernel/m/Read/ReadVariableOp$Adam/bn1/gamma/m/Read/ReadVariableOp#Adam/bn1/beta/m/Read/ReadVariableOp/Adam/binary_conv2d/kernel/m/Read/ReadVariableOp4Adam/batch_normalization/gamma/m/Read/ReadVariableOp3Adam/batch_normalization/beta/m/Read/ReadVariableOp(Adam/dense5/kernel/m/Read/ReadVariableOp$Adam/bn5/gamma/m/Read/ReadVariableOp#Adam/bn5/beta/m/Read/ReadVariableOp(Adam/dense6/kernel/m/Read/ReadVariableOp$Adam/bn6/gamma/m/Read/ReadVariableOp#Adam/bn6/beta/m/Read/ReadVariableOp'Adam/conv1/kernel/v/Read/ReadVariableOp$Adam/bn1/gamma/v/Read/ReadVariableOp#Adam/bn1/beta/v/Read/ReadVariableOp/Adam/binary_conv2d/kernel/v/Read/ReadVariableOp4Adam/batch_normalization/gamma/v/Read/ReadVariableOp3Adam/batch_normalization/beta/v/Read/ReadVariableOp(Adam/dense5/kernel/v/Read/ReadVariableOp$Adam/bn5/gamma/v/Read/ReadVariableOp#Adam/bn5/beta/v/Read/ReadVariableOp(Adam/dense6/kernel/v/Read/ReadVariableOp$Adam/bn6/gamma/v/Read/ReadVariableOp#Adam/bn6/beta/v/Read/ReadVariableOpConst*B
Tin;
927	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *'
f"R 
__inference__traced_save_28116
?

StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv1/kernel	bn1/gammabn1/betabn1/moving_meanbn1/moving_variancebinary_conv2d/kernelbatch_normalization/gammabatch_normalization/betabatch_normalization/moving_mean#batch_normalization/moving_variancedense5/kernel	bn5/gammabn5/betabn5/moving_meanbn5/moving_variancedense6/kernel	bn6/gammabn6/betabn6/moving_meanbn6/moving_variancebeta_1beta_2decaylearning_rate	Adam/itertotalcounttotal_1count_1Adam/conv1/kernel/mAdam/bn1/gamma/mAdam/bn1/beta/mAdam/binary_conv2d/kernel/m Adam/batch_normalization/gamma/mAdam/batch_normalization/beta/mAdam/dense5/kernel/mAdam/bn5/gamma/mAdam/bn5/beta/mAdam/dense6/kernel/mAdam/bn6/gamma/mAdam/bn6/beta/mAdam/conv1/kernel/vAdam/bn1/gamma/vAdam/bn1/beta/vAdam/binary_conv2d/kernel/v Adam/batch_normalization/gamma/vAdam/batch_normalization/beta/vAdam/dense5/kernel/vAdam/bn5/gamma/vAdam/bn5/beta/vAdam/dense6/kernel/vAdam/bn6/gamma/vAdam/bn6/beta/v*A
Tin:
826*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? **
f%R#
!__inference__traced_restore_28285??
?/
?
>__inference_bn6_layer_call_and_return_conditional_losses_26049

inputs
assignmovingavg_26024
assignmovingavg_1_26030)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identity??#AssignMovingAvg/AssignSubVariableOp?AssignMovingAvg/ReadVariableOp?%AssignMovingAvg_1/AssignSubVariableOp? AssignMovingAvg_1/ReadVariableOp?batchnorm/ReadVariableOp?batchnorm/mul/ReadVariableOp?
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indices?
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:d*
	keep_dims(2
moments/mean|
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:d2
moments/StopGradient?
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:?????????d2
moments/SquaredDifference?
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:d*
	keep_dims(2
moments/variance?
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:d*
squeeze_dims
 2
moments/Squeeze?
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:d*
squeeze_dims
 2
moments/Squeeze_1?
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*(
_class
loc:@AssignMovingAvg/26024*
_output_shapes
: *
dtype0*
valueB
 *???=2
AssignMovingAvg/decay?
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_26024*
_output_shapes
:d*
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*(
_class
loc:@AssignMovingAvg/26024*
_output_shapes
:d2
AssignMovingAvg/sub?
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*(
_class
loc:@AssignMovingAvg/26024*
_output_shapes
:d2
AssignMovingAvg/mul?
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_26024AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*(
_class
loc:@AssignMovingAvg/26024*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp?
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0**
_class 
loc:@AssignMovingAvg_1/26030*
_output_shapes
: *
dtype0*
valueB
 *???=2
AssignMovingAvg_1/decay?
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_26030*
_output_shapes
:d*
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0**
_class 
loc:@AssignMovingAvg_1/26030*
_output_shapes
:d2
AssignMovingAvg_1/sub?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0**
_class 
loc:@AssignMovingAvg_1/26030*
_output_shapes
:d2
AssignMovingAvg_1/mul?
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_26030AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0**
_class 
loc:@AssignMovingAvg_1/26030*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *?7?52
batchnorm/add/y?
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:d2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:d2
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:d*
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:d2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:?????????d2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:d2
batchnorm/mul_2?
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:d*
dtype02
batchnorm/ReadVariableOp?
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:d2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????d2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*'
_output_shapes
:?????????d2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????d::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
?
?
N__inference_batch_normalization_layer_call_and_return_conditional_losses_25802

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
data_formatNCHW*
epsilon%??'7*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?/
?
>__inference_bn5_layer_call_and_return_conditional_losses_25909

inputs
assignmovingavg_25884
assignmovingavg_1_25890)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identity??#AssignMovingAvg/AssignSubVariableOp?AssignMovingAvg/ReadVariableOp?%AssignMovingAvg_1/AssignSubVariableOp? AssignMovingAvg_1/ReadVariableOp?batchnorm/ReadVariableOp?batchnorm/mul/ReadVariableOp?
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indices?
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	?*
	keep_dims(2
moments/mean}
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	?2
moments/StopGradient?
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:??????????2
moments/SquaredDifference?
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	?*
	keep_dims(2
moments/variance?
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:?*
squeeze_dims
 2
moments/Squeeze?
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:?*
squeeze_dims
 2
moments/Squeeze_1?
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*(
_class
loc:@AssignMovingAvg/25884*
_output_shapes
: *
dtype0*
valueB
 *???=2
AssignMovingAvg/decay?
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_25884*
_output_shapes	
:?*
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*(
_class
loc:@AssignMovingAvg/25884*
_output_shapes	
:?2
AssignMovingAvg/sub?
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*(
_class
loc:@AssignMovingAvg/25884*
_output_shapes	
:?2
AssignMovingAvg/mul?
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_25884AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*(
_class
loc:@AssignMovingAvg/25884*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp?
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0**
_class 
loc:@AssignMovingAvg_1/25890*
_output_shapes
: *
dtype0*
valueB
 *???=2
AssignMovingAvg_1/decay?
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_25890*
_output_shapes	
:?*
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0**
_class 
loc:@AssignMovingAvg_1/25890*
_output_shapes	
:?2
AssignMovingAvg_1/sub?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0**
_class 
loc:@AssignMovingAvg_1/25890*
_output_shapes	
:?2
AssignMovingAvg_1/mul?
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_25890AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0**
_class 
loc:@AssignMovingAvg_1/25890*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *?7?52
batchnorm/add/y?
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:?2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:?2
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:?*
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:?2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:??????????2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:?2
batchnorm/mul_2?
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:?*
dtype02
batchnorm/ReadVariableOp?
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:??????????2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:??????????::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?<
?
E__inference_sequential_layer_call_and_return_conditional_losses_26720

inputs
conv1_26665
	bn1_26669
	bn1_26671
	bn1_26673
	bn1_26675
binary_conv2d_26679
batch_normalization_26683
batch_normalization_26685
batch_normalization_26687
batch_normalization_26689
dense5_26694
	bn5_26697
	bn5_26699
	bn5_26701
	bn5_26703
dense6_26707
	bn6_26710
	bn6_26712
	bn6_26714
	bn6_26716
identity??+batch_normalization/StatefulPartitionedCall?%binary_conv2d/StatefulPartitionedCall?bn1/StatefulPartitionedCall?bn5/StatefulPartitionedCall?bn6/StatefulPartitionedCall?conv1/StatefulPartitionedCall?dense5/StatefulPartitionedCall?dense6/StatefulPartitionedCall?
conv1/StatefulPartitionedCallStatefulPartitionedCallinputsconv1_26665*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@ *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_conv1_layer_call_and_return_conditional_losses_255372
conv1/StatefulPartitionedCall?
max_pooling2d/PartitionedCallPartitionedCall&conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????   * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_255512
max_pooling2d/PartitionedCall?
bn1/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0	bn1_26669	bn1_26671	bn1_26673	bn1_26675*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????   *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *G
fBR@
>__inference_bn1_layer_call_and_return_conditional_losses_261382
bn1/StatefulPartitionedCall?
act1/PartitionedCallPartitionedCall$bn1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????   * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *H
fCRA
?__inference_act1_layer_call_and_return_conditional_losses_261942
act1/PartitionedCall?
%binary_conv2d/StatefulPartitionedCallStatefulPartitionedCallact1/PartitionedCall:output:0binary_conv2d_26679*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????   *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_binary_conv2d_layer_call_and_return_conditional_losses_256892'
%binary_conv2d/StatefulPartitionedCall?
max_pooling2d_1/PartitionedCallPartitionedCall.binary_conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_257032!
max_pooling2d_1/PartitionedCall?
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_1/PartitionedCall:output:0batch_normalization_26683batch_normalization_26685batch_normalization_26687batch_normalization_26689*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_262432-
+batch_normalization/StatefulPartitionedCall?
activation/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_activation_layer_call_and_return_conditional_losses_262992
activation/PartitionedCall?
flatten/PartitionedCallPartitionedCall#activation/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_263132
flatten/PartitionedCall?
dense5/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense5_26694*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_dense5_layer_call_and_return_conditional_losses_263482 
dense5/StatefulPartitionedCall?
bn5/StatefulPartitionedCallStatefulPartitionedCall'dense5/StatefulPartitionedCall:output:0	bn5_26697	bn5_26699	bn5_26701	bn5_26703*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *G
fBR@
>__inference_bn5_layer_call_and_return_conditional_losses_259422
bn5/StatefulPartitionedCall?
act5/PartitionedCallPartitionedCall$bn5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *H
fCRA
?__inference_act5_layer_call_and_return_conditional_losses_264152
act5/PartitionedCall?
dense6/StatefulPartitionedCallStatefulPartitionedCallact5/PartitionedCall:output:0dense6_26707*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_dense6_layer_call_and_return_conditional_losses_264502 
dense6/StatefulPartitionedCall?
bn6/StatefulPartitionedCallStatefulPartitionedCall'dense6/StatefulPartitionedCall:output:0	bn6_26710	bn6_26712	bn6_26714	bn6_26716*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *G
fBR@
>__inference_bn6_layer_call_and_return_conditional_losses_260822
bn6/StatefulPartitionedCall?
IdentityIdentity$bn6/StatefulPartitionedCall:output:0,^batch_normalization/StatefulPartitionedCall&^binary_conv2d/StatefulPartitionedCall^bn1/StatefulPartitionedCall^bn5/StatefulPartitionedCall^bn6/StatefulPartitionedCall^conv1/StatefulPartitionedCall^dense5/StatefulPartitionedCall^dense6/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????d2

Identity"
identityIdentity:output:0*~
_input_shapesm
k:?????????@@::::::::::::::::::::2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2N
%binary_conv2d/StatefulPartitionedCall%binary_conv2d/StatefulPartitionedCall2:
bn1/StatefulPartitionedCallbn1/StatefulPartitionedCall2:
bn5/StatefulPartitionedCallbn5/StatefulPartitionedCall2:
bn6/StatefulPartitionedCallbn6/StatefulPartitionedCall2>
conv1/StatefulPartitionedCallconv1/StatefulPartitionedCall2@
dense5/StatefulPartitionedCalldense5/StatefulPartitionedCall2@
dense6/StatefulPartitionedCalldense6/StatefulPartitionedCall:W S
/
_output_shapes
:?????????@@
 
_user_specified_nameinputs
?
?
*__inference_sequential_layer_call_fn_26660
conv1_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallconv1_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18* 
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d*.
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_266172
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????d2

Identity"
identityIdentity:output:0*~
_input_shapesm
k:?????????@@::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:\ X
/
_output_shapes
:?????????@@
%
_user_specified_nameconv1_input
?
?
3__inference_batch_normalization_layer_call_fn_27564

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_262252
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:????????? ::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
A__inference_dense5_layer_call_and_return_conditional_losses_26348

inputs
readvariableop_resource
identity??ReadVariableOpz
ReadVariableOpReadVariableOpreadvariableop_resource* 
_output_shapes
:
?@?*
dtype02
ReadVariableOp[
	truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
	truediv/yt
truedivRealDivReadVariableOp:value:0truediv/y:output:0*
T0* 
_output_shapes
:
?@?2	
truedivS
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
mul/xY
mulMulmul/x:output:0truediv:z:0*
T0* 
_output_shapes
:
?@?2
mulS
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
add/yW
addAddV2mul:z:0add/y:output:0*
T0* 
_output_shapes
:
?@?2
addw
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value/Minimum/y?
clip_by_value/MinimumMinimumadd:z:0 clip_by_value/Minimum/y:output:0*
T0* 
_output_shapes
:
?@?2
clip_by_value/Minimumg
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value/y?
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0* 
_output_shapes
:
?@?2
clip_by_valueU
RoundRoundclip_by_value:z:0*
T0* 
_output_shapes
:
?@?2
RoundZ
subSub	Round:y:0clip_by_value:z:0*
T0* 
_output_shapes
:
?@?2
sub`
StopGradientStopGradientsub:z:0*
T0* 
_output_shapes
:
?@?2
StopGradientl
add_1AddV2clip_by_value:z:0StopGradient:output:0*
T0* 
_output_shapes
:
?@?2
add_1W
mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @2	
mul_1/x]
mul_1Mulmul_1/x:output:0	add_1:z:0*
T0* 
_output_shapes
:
?@?2
mul_1W
sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2	
sub_1/y]
sub_1Sub	mul_1:z:0sub_1/y:output:0*
T0* 
_output_shapes
:
?@?2
sub_1W
mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2	
mul_2/x]
mul_2Mulmul_2/x:output:0	sub_1:z:0*
T0* 
_output_shapes
:
?@?2
mul_2`
MatMulMatMulinputs	mul_2:z:0*
T0*(
_output_shapes
:??????????2
MatMulv
IdentityIdentityMatMul:product:0^ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*+
_input_shapes
:??????????@:2 
ReadVariableOpReadVariableOp:P L
(
_output_shapes
:??????????@
 
_user_specified_nameinputs
?
?
#__inference_bn6_layer_call_fn_27921

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *G
fBR@
>__inference_bn6_layer_call_and_return_conditional_losses_260492
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????d2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????d::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
?
?
>__inference_bn1_layer_call_and_return_conditional_losses_26138

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????   : : : : :*
data_formatNCHW*
epsilon%??'7*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:?????????   2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????   ::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????   
 
_user_specified_nameinputs
?
?
N__inference_batch_normalization_layer_call_and_return_conditional_losses_27597

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
data_formatNCHW*
epsilon%??'7*
exponential_avg_factor%???=2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
#__inference_bn1_layer_call_fn_27411

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+????????? ??????????????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *G
fBR@
>__inference_bn1_layer_call_and_return_conditional_losses_256192
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+????????? ??????????????????2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+????????? ??????????????????::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+????????? ??????????????????
 
_user_specified_nameinputs
?
?
N__inference_batch_normalization_layer_call_and_return_conditional_losses_27615

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
data_formatNCHW*
epsilon%??'7*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
[
?__inference_act5_layer_call_and_return_conditional_losses_27813

inputs
identityS
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
mul/x\
mulMulmul/x:output:0inputs*
T0*(
_output_shapes
:??????????2
mulS
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
add/y_
addAddV2mul:z:0add/y:output:0*
T0*(
_output_shapes
:??????????2
addw
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value/Minimum/y?
clip_by_value/MinimumMinimumadd:z:0 clip_by_value/Minimum/y:output:0*
T0*(
_output_shapes
:??????????2
clip_by_value/Minimumg
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value/y?
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*(
_output_shapes
:??????????2
clip_by_value]
RoundRoundclip_by_value:z:0*
T0*(
_output_shapes
:??????????2
Roundb
subSub	Round:y:0clip_by_value:z:0*
T0*(
_output_shapes
:??????????2
subh
StopGradientStopGradientsub:z:0*
T0*(
_output_shapes
:??????????2
StopGradientt
add_1AddV2clip_by_value:z:0StopGradient:output:0*
T0*(
_output_shapes
:??????????2
add_1W
mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @2	
mul_1/xe
mul_1Mulmul_1/x:output:0	add_1:z:0*
T0*(
_output_shapes
:??????????2
mul_1W
sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2	
sub_1/ye
sub_1Sub	mul_1:z:0sub_1/y:output:0*
T0*(
_output_shapes
:??????????2
sub_1^
IdentityIdentity	sub_1:z:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
#__inference_bn6_layer_call_fn_27934

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *G
fBR@
>__inference_bn6_layer_call_and_return_conditional_losses_260822
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????d2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????d::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
?
?
>__inference_bn1_layer_call_and_return_conditional_losses_25650

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+????????? ??????????????????: : : : :*
data_formatNCHW*
epsilon%??'7*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+????????? ??????????????????2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+????????? ??????????????????::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+????????? ??????????????????
 
_user_specified_nameinputs
?
a
E__inference_activation_layer_call_and_return_conditional_losses_27661

inputs
identityS
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
mul/xc
mulMulmul/x:output:0inputs*
T0*/
_output_shapes
:????????? 2
mulS
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
add/yf
addAddV2mul:z:0add/y:output:0*
T0*/
_output_shapes
:????????? 2
addw
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value/Minimum/y?
clip_by_value/MinimumMinimumadd:z:0 clip_by_value/Minimum/y:output:0*
T0*/
_output_shapes
:????????? 2
clip_by_value/Minimumg
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value/y?
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*/
_output_shapes
:????????? 2
clip_by_valued
RoundRoundclip_by_value:z:0*
T0*/
_output_shapes
:????????? 2
Roundi
subSub	Round:y:0clip_by_value:z:0*
T0*/
_output_shapes
:????????? 2
subo
StopGradientStopGradientsub:z:0*
T0*/
_output_shapes
:????????? 2
StopGradient{
add_1AddV2clip_by_value:z:0StopGradient:output:0*
T0*/
_output_shapes
:????????? 2
add_1W
mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @2	
mul_1/xl
mul_1Mulmul_1/x:output:0	add_1:z:0*
T0*/
_output_shapes
:????????? 2
mul_1W
sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2	
sub_1/yl
sub_1Sub	mul_1:z:0sub_1/y:output:0*
T0*/
_output_shapes
:????????? 2
sub_1e
IdentityIdentity	sub_1:z:0*
T0*/
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*.
_input_shapes
:????????? :W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
A__inference_dense6_layer_call_and_return_conditional_losses_27845

inputs
readvariableop_resource
identity??ReadVariableOpy
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:	?d*
dtype02
ReadVariableOp[
	truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
	truediv/ys
truedivRealDivReadVariableOp:value:0truediv/y:output:0*
T0*
_output_shapes
:	?d2	
truedivS
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
mul/xX
mulMulmul/x:output:0truediv:z:0*
T0*
_output_shapes
:	?d2
mulS
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
add/yV
addAddV2mul:z:0add/y:output:0*
T0*
_output_shapes
:	?d2
addw
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value/Minimum/y?
clip_by_value/MinimumMinimumadd:z:0 clip_by_value/Minimum/y:output:0*
T0*
_output_shapes
:	?d2
clip_by_value/Minimumg
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value/y?
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*
_output_shapes
:	?d2
clip_by_valueT
RoundRoundclip_by_value:z:0*
T0*
_output_shapes
:	?d2
RoundY
subSub	Round:y:0clip_by_value:z:0*
T0*
_output_shapes
:	?d2
sub_
StopGradientStopGradientsub:z:0*
T0*
_output_shapes
:	?d2
StopGradientk
add_1AddV2clip_by_value:z:0StopGradient:output:0*
T0*
_output_shapes
:	?d2
add_1W
mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @2	
mul_1/x\
mul_1Mulmul_1/x:output:0	add_1:z:0*
T0*
_output_shapes
:	?d2
mul_1W
sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2	
sub_1/y\
sub_1Sub	mul_1:z:0sub_1/y:output:0*
T0*
_output_shapes
:	?d2
sub_1W
mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2	
mul_2/x\
mul_2Mulmul_2/x:output:0	sub_1:z:0*
T0*
_output_shapes
:	?d2
mul_2_
MatMulMatMulinputs	mul_2:z:0*
T0*'
_output_shapes
:?????????d2
MatMulu
IdentityIdentityMatMul:product:0^ReadVariableOp*
T0*'
_output_shapes
:?????????d2

Identity"
identityIdentity:output:0*+
_input_shapes
:??????????:2 
ReadVariableOpReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?/
?
>__inference_bn6_layer_call_and_return_conditional_losses_27888

inputs
assignmovingavg_27863
assignmovingavg_1_27869)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identity??#AssignMovingAvg/AssignSubVariableOp?AssignMovingAvg/ReadVariableOp?%AssignMovingAvg_1/AssignSubVariableOp? AssignMovingAvg_1/ReadVariableOp?batchnorm/ReadVariableOp?batchnorm/mul/ReadVariableOp?
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indices?
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:d*
	keep_dims(2
moments/mean|
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:d2
moments/StopGradient?
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:?????????d2
moments/SquaredDifference?
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:d*
	keep_dims(2
moments/variance?
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:d*
squeeze_dims
 2
moments/Squeeze?
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:d*
squeeze_dims
 2
moments/Squeeze_1?
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*(
_class
loc:@AssignMovingAvg/27863*
_output_shapes
: *
dtype0*
valueB
 *???=2
AssignMovingAvg/decay?
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_27863*
_output_shapes
:d*
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*(
_class
loc:@AssignMovingAvg/27863*
_output_shapes
:d2
AssignMovingAvg/sub?
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*(
_class
loc:@AssignMovingAvg/27863*
_output_shapes
:d2
AssignMovingAvg/mul?
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_27863AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*(
_class
loc:@AssignMovingAvg/27863*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp?
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0**
_class 
loc:@AssignMovingAvg_1/27869*
_output_shapes
: *
dtype0*
valueB
 *???=2
AssignMovingAvg_1/decay?
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_27869*
_output_shapes
:d*
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0**
_class 
loc:@AssignMovingAvg_1/27869*
_output_shapes
:d2
AssignMovingAvg_1/sub?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0**
_class 
loc:@AssignMovingAvg_1/27869*
_output_shapes
:d2
AssignMovingAvg_1/mul?
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_27869AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0**
_class 
loc:@AssignMovingAvg_1/27869*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *?7?52
batchnorm/add/y?
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:d2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:d2
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:d*
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:d2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:?????????d2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:d2
batchnorm/mul_2?
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:d*
dtype02
batchnorm/ReadVariableOp?
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:d2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????d2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*'
_output_shapes
:?????????d2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????d::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
?
?
3__inference_batch_normalization_layer_call_fn_27628

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_257712
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
>__inference_bn5_layer_call_and_return_conditional_losses_27767

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identity??batchnorm/ReadVariableOp?batchnorm/ReadVariableOp_1?batchnorm/ReadVariableOp_2?batchnorm/mul/ReadVariableOp?
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:?*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *?7?52
batchnorm/add/y?
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:?2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:?2
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:?*
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:?2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:??????????2
batchnorm/mul_1?
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:?*
dtype02
batchnorm/ReadVariableOp_1?
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:?2
batchnorm/mul_2?
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:?*
dtype02
batchnorm/ReadVariableOp_2?
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:??????????2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:??????????::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
??
?
!__inference__traced_restore_28285
file_prefix!
assignvariableop_conv1_kernel 
assignvariableop_1_bn1_gamma
assignvariableop_2_bn1_beta&
"assignvariableop_3_bn1_moving_mean*
&assignvariableop_4_bn1_moving_variance+
'assignvariableop_5_binary_conv2d_kernel0
,assignvariableop_6_batch_normalization_gamma/
+assignvariableop_7_batch_normalization_beta6
2assignvariableop_8_batch_normalization_moving_mean:
6assignvariableop_9_batch_normalization_moving_variance%
!assignvariableop_10_dense5_kernel!
assignvariableop_11_bn5_gamma 
assignvariableop_12_bn5_beta'
#assignvariableop_13_bn5_moving_mean+
'assignvariableop_14_bn5_moving_variance%
!assignvariableop_15_dense6_kernel!
assignvariableop_16_bn6_gamma 
assignvariableop_17_bn6_beta'
#assignvariableop_18_bn6_moving_mean+
'assignvariableop_19_bn6_moving_variance
assignvariableop_20_beta_1
assignvariableop_21_beta_2
assignvariableop_22_decay%
!assignvariableop_23_learning_rate!
assignvariableop_24_adam_iter
assignvariableop_25_total
assignvariableop_26_count
assignvariableop_27_total_1
assignvariableop_28_count_1+
'assignvariableop_29_adam_conv1_kernel_m(
$assignvariableop_30_adam_bn1_gamma_m'
#assignvariableop_31_adam_bn1_beta_m3
/assignvariableop_32_adam_binary_conv2d_kernel_m8
4assignvariableop_33_adam_batch_normalization_gamma_m7
3assignvariableop_34_adam_batch_normalization_beta_m,
(assignvariableop_35_adam_dense5_kernel_m(
$assignvariableop_36_adam_bn5_gamma_m'
#assignvariableop_37_adam_bn5_beta_m,
(assignvariableop_38_adam_dense6_kernel_m(
$assignvariableop_39_adam_bn6_gamma_m'
#assignvariableop_40_adam_bn6_beta_m+
'assignvariableop_41_adam_conv1_kernel_v(
$assignvariableop_42_adam_bn1_gamma_v'
#assignvariableop_43_adam_bn1_beta_v3
/assignvariableop_44_adam_binary_conv2d_kernel_v8
4assignvariableop_45_adam_batch_normalization_gamma_v7
3assignvariableop_46_adam_batch_normalization_beta_v,
(assignvariableop_47_adam_dense5_kernel_v(
$assignvariableop_48_adam_bn5_gamma_v'
#assignvariableop_49_adam_bn5_beta_v,
(assignvariableop_50_adam_dense6_kernel_v(
$assignvariableop_51_adam_bn6_gamma_v'
#assignvariableop_52_adam_bn6_beta_v
identity_54??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_37?AssignVariableOp_38?AssignVariableOp_39?AssignVariableOp_4?AssignVariableOp_40?AssignVariableOp_41?AssignVariableOp_42?AssignVariableOp_43?AssignVariableOp_44?AssignVariableOp_45?AssignVariableOp_46?AssignVariableOp_47?AssignVariableOp_48?AssignVariableOp_49?AssignVariableOp_5?AssignVariableOp_50?AssignVariableOp_51?AssignVariableOp_52?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:6*
dtype0*?
value?B?6B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:6*
dtype0*
valuevBt6B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?::::::::::::::::::::::::::::::::::::::::::::::::::::::*D
dtypes:
826	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOpassignvariableop_conv1_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOpassignvariableop_1_bn1_gammaIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOpassignvariableop_2_bn1_betaIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp"assignvariableop_3_bn1_moving_meanIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp&assignvariableop_4_bn1_moving_varianceIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp'assignvariableop_5_binary_conv2d_kernelIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp,assignvariableop_6_batch_normalization_gammaIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOp+assignvariableop_7_batch_normalization_betaIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp2assignvariableop_8_batch_normalization_moving_meanIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOp6assignvariableop_9_batch_normalization_moving_varianceIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp!assignvariableop_10_dense5_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOpassignvariableop_11_bn5_gammaIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOpassignvariableop_12_bn5_betaIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOp#assignvariableop_13_bn5_moving_meanIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOp'assignvariableop_14_bn5_moving_varianceIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOp!assignvariableop_15_dense6_kernelIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOpassignvariableop_16_bn6_gammaIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOpassignvariableop_17_bn6_betaIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOp#assignvariableop_18_bn6_moving_meanIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOp'assignvariableop_19_bn6_moving_varianceIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOpassignvariableop_20_beta_1Identity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOpassignvariableop_21_beta_2Identity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOpassignvariableop_22_decayIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOp!assignvariableop_23_learning_rateIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOpassignvariableop_24_adam_iterIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOpassignvariableop_25_totalIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOpassignvariableop_26_countIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27?
AssignVariableOp_27AssignVariableOpassignvariableop_27_total_1Identity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28?
AssignVariableOp_28AssignVariableOpassignvariableop_28_count_1Identity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29?
AssignVariableOp_29AssignVariableOp'assignvariableop_29_adam_conv1_kernel_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30?
AssignVariableOp_30AssignVariableOp$assignvariableop_30_adam_bn1_gamma_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31?
AssignVariableOp_31AssignVariableOp#assignvariableop_31_adam_bn1_beta_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32?
AssignVariableOp_32AssignVariableOp/assignvariableop_32_adam_binary_conv2d_kernel_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33?
AssignVariableOp_33AssignVariableOp4assignvariableop_33_adam_batch_normalization_gamma_mIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34?
AssignVariableOp_34AssignVariableOp3assignvariableop_34_adam_batch_normalization_beta_mIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35?
AssignVariableOp_35AssignVariableOp(assignvariableop_35_adam_dense5_kernel_mIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36?
AssignVariableOp_36AssignVariableOp$assignvariableop_36_adam_bn5_gamma_mIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37?
AssignVariableOp_37AssignVariableOp#assignvariableop_37_adam_bn5_beta_mIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38?
AssignVariableOp_38AssignVariableOp(assignvariableop_38_adam_dense6_kernel_mIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39?
AssignVariableOp_39AssignVariableOp$assignvariableop_39_adam_bn6_gamma_mIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40?
AssignVariableOp_40AssignVariableOp#assignvariableop_40_adam_bn6_beta_mIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41?
AssignVariableOp_41AssignVariableOp'assignvariableop_41_adam_conv1_kernel_vIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42?
AssignVariableOp_42AssignVariableOp$assignvariableop_42_adam_bn1_gamma_vIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43?
AssignVariableOp_43AssignVariableOp#assignvariableop_43_adam_bn1_beta_vIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44?
AssignVariableOp_44AssignVariableOp/assignvariableop_44_adam_binary_conv2d_kernel_vIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45?
AssignVariableOp_45AssignVariableOp4assignvariableop_45_adam_batch_normalization_gamma_vIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46?
AssignVariableOp_46AssignVariableOp3assignvariableop_46_adam_batch_normalization_beta_vIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47?
AssignVariableOp_47AssignVariableOp(assignvariableop_47_adam_dense5_kernel_vIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48?
AssignVariableOp_48AssignVariableOp$assignvariableop_48_adam_bn5_gamma_vIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49?
AssignVariableOp_49AssignVariableOp#assignvariableop_49_adam_bn5_beta_vIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50?
AssignVariableOp_50AssignVariableOp(assignvariableop_50_adam_dense6_kernel_vIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51?
AssignVariableOp_51AssignVariableOp$assignvariableop_51_adam_bn6_gamma_vIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_51n
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:2
Identity_52?
AssignVariableOp_52AssignVariableOp#assignvariableop_52_adam_bn6_beta_vIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_529
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?	
Identity_53Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_53?	
Identity_54IdentityIdentity_53:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_54"#
identity_54Identity_54:output:0*?
_input_shapes?
?: :::::::::::::::::::::::::::::::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?
?
#__inference_signature_wrapper_26818
conv1_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallconv1_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18* 
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d*6
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *)
f$R"
 __inference__wrapped_model_255092
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????d2

Identity"
identityIdentity:output:0*~
_input_shapesm
k:?????????@@::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:\ X
/
_output_shapes
:?????????@@
%
_user_specified_nameconv1_input
?
?
#__inference_bn1_layer_call_fn_27488

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????   *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *G
fBR@
>__inference_bn1_layer_call_and_return_conditional_losses_261382
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????   2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????   ::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????   
 
_user_specified_nameinputs
?
[
?__inference_act1_layer_call_and_return_conditional_losses_27508

inputs
identityS
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
mul/xc
mulMulmul/x:output:0inputs*
T0*/
_output_shapes
:?????????   2
mulS
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
add/yf
addAddV2mul:z:0add/y:output:0*
T0*/
_output_shapes
:?????????   2
addw
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value/Minimum/y?
clip_by_value/MinimumMinimumadd:z:0 clip_by_value/Minimum/y:output:0*
T0*/
_output_shapes
:?????????   2
clip_by_value/Minimumg
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value/y?
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*/
_output_shapes
:?????????   2
clip_by_valued
RoundRoundclip_by_value:z:0*
T0*/
_output_shapes
:?????????   2
Roundi
subSub	Round:y:0clip_by_value:z:0*
T0*/
_output_shapes
:?????????   2
subo
StopGradientStopGradientsub:z:0*
T0*/
_output_shapes
:?????????   2
StopGradient{
add_1AddV2clip_by_value:z:0StopGradient:output:0*
T0*/
_output_shapes
:?????????   2
add_1W
mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @2	
mul_1/xl
mul_1Mulmul_1/x:output:0	add_1:z:0*
T0*/
_output_shapes
:?????????   2
mul_1W
sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2	
sub_1/yl
sub_1Sub	mul_1:z:0sub_1/y:output:0*
T0*/
_output_shapes
:?????????   2
sub_1e
IdentityIdentity	sub_1:z:0*
T0*/
_output_shapes
:?????????   2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????   :W S
/
_output_shapes
:?????????   
 
_user_specified_nameinputs
?
@
$__inference_act1_layer_call_fn_27513

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????   * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *H
fCRA
?__inference_act1_layer_call_and_return_conditional_losses_261942
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????   2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????   :W S
/
_output_shapes
:?????????   
 
_user_specified_nameinputs
?
?
N__inference_batch_normalization_layer_call_and_return_conditional_losses_27551

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:????????? :::::*
data_formatNCHW*
epsilon%??'7*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:????????? ::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
I
-__inference_max_pooling2d_layer_call_fn_25557

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_255512
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
f
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_25703

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
#__inference_bn5_layer_call_fn_27793

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *G
fBR@
>__inference_bn5_layer_call_and_return_conditional_losses_259422
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:??????????::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
[
?__inference_act5_layer_call_and_return_conditional_losses_26415

inputs
identityS
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
mul/x\
mulMulmul/x:output:0inputs*
T0*(
_output_shapes
:??????????2
mulS
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
add/y_
addAddV2mul:z:0add/y:output:0*
T0*(
_output_shapes
:??????????2
addw
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value/Minimum/y?
clip_by_value/MinimumMinimumadd:z:0 clip_by_value/Minimum/y:output:0*
T0*(
_output_shapes
:??????????2
clip_by_value/Minimumg
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value/y?
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*(
_output_shapes
:??????????2
clip_by_value]
RoundRoundclip_by_value:z:0*
T0*(
_output_shapes
:??????????2
Roundb
subSub	Round:y:0clip_by_value:z:0*
T0*(
_output_shapes
:??????????2
subh
StopGradientStopGradientsub:z:0*
T0*(
_output_shapes
:??????????2
StopGradientt
add_1AddV2clip_by_value:z:0StopGradient:output:0*
T0*(
_output_shapes
:??????????2
add_1W
mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @2	
mul_1/xe
mul_1Mulmul_1/x:output:0	add_1:z:0*
T0*(
_output_shapes
:??????????2
mul_1W
sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2	
sub_1/ye
sub_1Sub	mul_1:z:0sub_1/y:output:0*
T0*(
_output_shapes
:??????????2
sub_1^
IdentityIdentity	sub_1:z:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
>__inference_bn1_layer_call_and_return_conditional_losses_27462

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????   : : : : :*
data_formatNCHW*
epsilon%??'7*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:?????????   2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????   ::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????   
 
_user_specified_nameinputs
?
?
>__inference_bn1_layer_call_and_return_conditional_losses_27398

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+????????? ??????????????????: : : : :*
data_formatNCHW*
epsilon%??'7*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+????????? ??????????????????2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+????????? ??????????????????::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+????????? ??????????????????
 
_user_specified_nameinputs
?
?
#__inference_bn1_layer_call_fn_27424

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+????????? ??????????????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *G
fBR@
>__inference_bn1_layer_call_and_return_conditional_losses_256502
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+????????? ??????????????????2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+????????? ??????????????????::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+????????? ??????????????????
 
_user_specified_nameinputs
?
?
#__inference_bn1_layer_call_fn_27475

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????   *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *G
fBR@
>__inference_bn1_layer_call_and_return_conditional_losses_261202
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????   2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????   ::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????   
 
_user_specified_nameinputs
?
l
&__inference_dense5_layer_call_fn_27711

inputs
unknown
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_dense5_layer_call_and_return_conditional_losses_263482
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*+
_input_shapes
:??????????@:22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????@
 
_user_specified_nameinputs
?
?
N__inference_batch_normalization_layer_call_and_return_conditional_losses_27533

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:????????? :::::*
data_formatNCHW*
epsilon%??'7*
exponential_avg_factor%???=2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:????????? ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
A__inference_dense6_layer_call_and_return_conditional_losses_26450

inputs
readvariableop_resource
identity??ReadVariableOpy
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:	?d*
dtype02
ReadVariableOp[
	truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
	truediv/ys
truedivRealDivReadVariableOp:value:0truediv/y:output:0*
T0*
_output_shapes
:	?d2	
truedivS
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
mul/xX
mulMulmul/x:output:0truediv:z:0*
T0*
_output_shapes
:	?d2
mulS
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
add/yV
addAddV2mul:z:0add/y:output:0*
T0*
_output_shapes
:	?d2
addw
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value/Minimum/y?
clip_by_value/MinimumMinimumadd:z:0 clip_by_value/Minimum/y:output:0*
T0*
_output_shapes
:	?d2
clip_by_value/Minimumg
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value/y?
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*
_output_shapes
:	?d2
clip_by_valueT
RoundRoundclip_by_value:z:0*
T0*
_output_shapes
:	?d2
RoundY
subSub	Round:y:0clip_by_value:z:0*
T0*
_output_shapes
:	?d2
sub_
StopGradientStopGradientsub:z:0*
T0*
_output_shapes
:	?d2
StopGradientk
add_1AddV2clip_by_value:z:0StopGradient:output:0*
T0*
_output_shapes
:	?d2
add_1W
mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @2	
mul_1/x\
mul_1Mulmul_1/x:output:0	add_1:z:0*
T0*
_output_shapes
:	?d2
mul_1W
sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2	
sub_1/y\
sub_1Sub	mul_1:z:0sub_1/y:output:0*
T0*
_output_shapes
:	?d2
sub_1W
mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2	
mul_2/x\
mul_2Mulmul_2/x:output:0	sub_1:z:0*
T0*
_output_shapes
:	?d2
mul_2_
MatMulMatMulinputs	mul_2:z:0*
T0*'
_output_shapes
:?????????d2
MatMulu
IdentityIdentityMatMul:product:0^ReadVariableOp*
T0*'
_output_shapes
:?????????d2

Identity"
identityIdentity:output:0*+
_input_shapes
:??????????:2 
ReadVariableOpReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
#__inference_bn5_layer_call_fn_27780

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *G
fBR@
>__inference_bn5_layer_call_and_return_conditional_losses_259092
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:??????????::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
F
*__inference_activation_layer_call_fn_27666

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_activation_layer_call_and_return_conditional_losses_262992
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*.
_input_shapes
:????????? :W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
C
'__inference_flatten_layer_call_fn_27677

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_263132
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????@2

Identity"
identityIdentity:output:0*.
_input_shapes
:????????? :W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?<
?
E__inference_sequential_layer_call_and_return_conditional_losses_26498
conv1_input
conv1_26097
	bn1_26165
	bn1_26167
	bn1_26169
	bn1_26171
binary_conv2d_26202
batch_normalization_26270
batch_normalization_26272
batch_normalization_26274
batch_normalization_26276
dense5_26357
	bn5_26386
	bn5_26388
	bn5_26390
	bn5_26392
dense6_26459
	bn6_26488
	bn6_26490
	bn6_26492
	bn6_26494
identity??+batch_normalization/StatefulPartitionedCall?%binary_conv2d/StatefulPartitionedCall?bn1/StatefulPartitionedCall?bn5/StatefulPartitionedCall?bn6/StatefulPartitionedCall?conv1/StatefulPartitionedCall?dense5/StatefulPartitionedCall?dense6/StatefulPartitionedCall?
conv1/StatefulPartitionedCallStatefulPartitionedCallconv1_inputconv1_26097*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@ *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_conv1_layer_call_and_return_conditional_losses_255372
conv1/StatefulPartitionedCall?
max_pooling2d/PartitionedCallPartitionedCall&conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????   * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_255512
max_pooling2d/PartitionedCall?
bn1/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0	bn1_26165	bn1_26167	bn1_26169	bn1_26171*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????   *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *G
fBR@
>__inference_bn1_layer_call_and_return_conditional_losses_261202
bn1/StatefulPartitionedCall?
act1/PartitionedCallPartitionedCall$bn1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????   * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *H
fCRA
?__inference_act1_layer_call_and_return_conditional_losses_261942
act1/PartitionedCall?
%binary_conv2d/StatefulPartitionedCallStatefulPartitionedCallact1/PartitionedCall:output:0binary_conv2d_26202*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????   *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_binary_conv2d_layer_call_and_return_conditional_losses_256892'
%binary_conv2d/StatefulPartitionedCall?
max_pooling2d_1/PartitionedCallPartitionedCall.binary_conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_257032!
max_pooling2d_1/PartitionedCall?
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_1/PartitionedCall:output:0batch_normalization_26270batch_normalization_26272batch_normalization_26274batch_normalization_26276*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_262252-
+batch_normalization/StatefulPartitionedCall?
activation/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_activation_layer_call_and_return_conditional_losses_262992
activation/PartitionedCall?
flatten/PartitionedCallPartitionedCall#activation/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_263132
flatten/PartitionedCall?
dense5/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense5_26357*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_dense5_layer_call_and_return_conditional_losses_263482 
dense5/StatefulPartitionedCall?
bn5/StatefulPartitionedCallStatefulPartitionedCall'dense5/StatefulPartitionedCall:output:0	bn5_26386	bn5_26388	bn5_26390	bn5_26392*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *G
fBR@
>__inference_bn5_layer_call_and_return_conditional_losses_259092
bn5/StatefulPartitionedCall?
act5/PartitionedCallPartitionedCall$bn5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *H
fCRA
?__inference_act5_layer_call_and_return_conditional_losses_264152
act5/PartitionedCall?
dense6/StatefulPartitionedCallStatefulPartitionedCallact5/PartitionedCall:output:0dense6_26459*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_dense6_layer_call_and_return_conditional_losses_264502 
dense6/StatefulPartitionedCall?
bn6/StatefulPartitionedCallStatefulPartitionedCall'dense6/StatefulPartitionedCall:output:0	bn6_26488	bn6_26490	bn6_26492	bn6_26494*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *G
fBR@
>__inference_bn6_layer_call_and_return_conditional_losses_260492
bn6/StatefulPartitionedCall?
IdentityIdentity$bn6/StatefulPartitionedCall:output:0,^batch_normalization/StatefulPartitionedCall&^binary_conv2d/StatefulPartitionedCall^bn1/StatefulPartitionedCall^bn5/StatefulPartitionedCall^bn6/StatefulPartitionedCall^conv1/StatefulPartitionedCall^dense5/StatefulPartitionedCall^dense6/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????d2

Identity"
identityIdentity:output:0*~
_input_shapesm
k:?????????@@::::::::::::::::::::2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2N
%binary_conv2d/StatefulPartitionedCall%binary_conv2d/StatefulPartitionedCall2:
bn1/StatefulPartitionedCallbn1/StatefulPartitionedCall2:
bn5/StatefulPartitionedCallbn5/StatefulPartitionedCall2:
bn6/StatefulPartitionedCallbn6/StatefulPartitionedCall2>
conv1/StatefulPartitionedCallconv1/StatefulPartitionedCall2@
dense5/StatefulPartitionedCalldense5/StatefulPartitionedCall2@
dense6/StatefulPartitionedCalldense6/StatefulPartitionedCall:\ X
/
_output_shapes
:?????????@@
%
_user_specified_nameconv1_input
?
^
B__inference_flatten_layer_call_and_return_conditional_losses_26313

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????    2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????@2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????@2

Identity"
identityIdentity:output:0*.
_input_shapes
:????????? :W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
3__inference_batch_normalization_layer_call_fn_27577

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_262432
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:????????? ::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
a
E__inference_activation_layer_call_and_return_conditional_losses_26299

inputs
identityS
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
mul/xc
mulMulmul/x:output:0inputs*
T0*/
_output_shapes
:????????? 2
mulS
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
add/yf
addAddV2mul:z:0add/y:output:0*
T0*/
_output_shapes
:????????? 2
addw
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value/Minimum/y?
clip_by_value/MinimumMinimumadd:z:0 clip_by_value/Minimum/y:output:0*
T0*/
_output_shapes
:????????? 2
clip_by_value/Minimumg
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value/y?
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*/
_output_shapes
:????????? 2
clip_by_valued
RoundRoundclip_by_value:z:0*
T0*/
_output_shapes
:????????? 2
Roundi
subSub	Round:y:0clip_by_value:z:0*
T0*/
_output_shapes
:????????? 2
subo
StopGradientStopGradientsub:z:0*
T0*/
_output_shapes
:????????? 2
StopGradient{
add_1AddV2clip_by_value:z:0StopGradient:output:0*
T0*/
_output_shapes
:????????? 2
add_1W
mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @2	
mul_1/xl
mul_1Mulmul_1/x:output:0	add_1:z:0*
T0*/
_output_shapes
:????????? 2
mul_1W
sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2	
sub_1/yl
sub_1Sub	mul_1:z:0sub_1/y:output:0*
T0*/
_output_shapes
:????????? 2
sub_1e
IdentityIdentity	sub_1:z:0*
T0*/
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*.
_input_shapes
:????????? :W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
N__inference_batch_normalization_layer_call_and_return_conditional_losses_25771

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
data_formatNCHW*
epsilon%??'7*
exponential_avg_factor%???=2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
[
?__inference_act1_layer_call_and_return_conditional_losses_26194

inputs
identityS
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
mul/xc
mulMulmul/x:output:0inputs*
T0*/
_output_shapes
:?????????   2
mulS
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
add/yf
addAddV2mul:z:0add/y:output:0*
T0*/
_output_shapes
:?????????   2
addw
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value/Minimum/y?
clip_by_value/MinimumMinimumadd:z:0 clip_by_value/Minimum/y:output:0*
T0*/
_output_shapes
:?????????   2
clip_by_value/Minimumg
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value/y?
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*/
_output_shapes
:?????????   2
clip_by_valued
RoundRoundclip_by_value:z:0*
T0*/
_output_shapes
:?????????   2
Roundi
subSub	Round:y:0clip_by_value:z:0*
T0*/
_output_shapes
:?????????   2
subo
StopGradientStopGradientsub:z:0*
T0*/
_output_shapes
:?????????   2
StopGradient{
add_1AddV2clip_by_value:z:0StopGradient:output:0*
T0*/
_output_shapes
:?????????   2
add_1W
mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @2	
mul_1/xl
mul_1Mulmul_1/x:output:0	add_1:z:0*
T0*/
_output_shapes
:?????????   2
mul_1W
sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2	
sub_1/yl
sub_1Sub	mul_1:z:0sub_1/y:output:0*
T0*/
_output_shapes
:?????????   2
sub_1e
IdentityIdentity	sub_1:z:0*
T0*/
_output_shapes
:?????????   2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????   :W S
/
_output_shapes
:?????????   
 
_user_specified_nameinputs
?
?
N__inference_batch_normalization_layer_call_and_return_conditional_losses_26225

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:????????? :::::*
data_formatNCHW*
epsilon%??'7*
exponential_avg_factor%???=2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:????????? ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?<
?
E__inference_sequential_layer_call_and_return_conditional_losses_26556
conv1_input
conv1_26501
	bn1_26505
	bn1_26507
	bn1_26509
	bn1_26511
binary_conv2d_26515
batch_normalization_26519
batch_normalization_26521
batch_normalization_26523
batch_normalization_26525
dense5_26530
	bn5_26533
	bn5_26535
	bn5_26537
	bn5_26539
dense6_26543
	bn6_26546
	bn6_26548
	bn6_26550
	bn6_26552
identity??+batch_normalization/StatefulPartitionedCall?%binary_conv2d/StatefulPartitionedCall?bn1/StatefulPartitionedCall?bn5/StatefulPartitionedCall?bn6/StatefulPartitionedCall?conv1/StatefulPartitionedCall?dense5/StatefulPartitionedCall?dense6/StatefulPartitionedCall?
conv1/StatefulPartitionedCallStatefulPartitionedCallconv1_inputconv1_26501*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@ *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_conv1_layer_call_and_return_conditional_losses_255372
conv1/StatefulPartitionedCall?
max_pooling2d/PartitionedCallPartitionedCall&conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????   * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_255512
max_pooling2d/PartitionedCall?
bn1/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0	bn1_26505	bn1_26507	bn1_26509	bn1_26511*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????   *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *G
fBR@
>__inference_bn1_layer_call_and_return_conditional_losses_261382
bn1/StatefulPartitionedCall?
act1/PartitionedCallPartitionedCall$bn1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????   * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *H
fCRA
?__inference_act1_layer_call_and_return_conditional_losses_261942
act1/PartitionedCall?
%binary_conv2d/StatefulPartitionedCallStatefulPartitionedCallact1/PartitionedCall:output:0binary_conv2d_26515*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????   *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_binary_conv2d_layer_call_and_return_conditional_losses_256892'
%binary_conv2d/StatefulPartitionedCall?
max_pooling2d_1/PartitionedCallPartitionedCall.binary_conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_257032!
max_pooling2d_1/PartitionedCall?
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_1/PartitionedCall:output:0batch_normalization_26519batch_normalization_26521batch_normalization_26523batch_normalization_26525*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_262432-
+batch_normalization/StatefulPartitionedCall?
activation/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_activation_layer_call_and_return_conditional_losses_262992
activation/PartitionedCall?
flatten/PartitionedCallPartitionedCall#activation/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_263132
flatten/PartitionedCall?
dense5/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense5_26530*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_dense5_layer_call_and_return_conditional_losses_263482 
dense5/StatefulPartitionedCall?
bn5/StatefulPartitionedCallStatefulPartitionedCall'dense5/StatefulPartitionedCall:output:0	bn5_26533	bn5_26535	bn5_26537	bn5_26539*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *G
fBR@
>__inference_bn5_layer_call_and_return_conditional_losses_259422
bn5/StatefulPartitionedCall?
act5/PartitionedCallPartitionedCall$bn5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *H
fCRA
?__inference_act5_layer_call_and_return_conditional_losses_264152
act5/PartitionedCall?
dense6/StatefulPartitionedCallStatefulPartitionedCallact5/PartitionedCall:output:0dense6_26543*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_dense6_layer_call_and_return_conditional_losses_264502 
dense6/StatefulPartitionedCall?
bn6/StatefulPartitionedCallStatefulPartitionedCall'dense6/StatefulPartitionedCall:output:0	bn6_26546	bn6_26548	bn6_26550	bn6_26552*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *G
fBR@
>__inference_bn6_layer_call_and_return_conditional_losses_260822
bn6/StatefulPartitionedCall?
IdentityIdentity$bn6/StatefulPartitionedCall:output:0,^batch_normalization/StatefulPartitionedCall&^binary_conv2d/StatefulPartitionedCall^bn1/StatefulPartitionedCall^bn5/StatefulPartitionedCall^bn6/StatefulPartitionedCall^conv1/StatefulPartitionedCall^dense5/StatefulPartitionedCall^dense6/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????d2

Identity"
identityIdentity:output:0*~
_input_shapesm
k:?????????@@::::::::::::::::::::2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2N
%binary_conv2d/StatefulPartitionedCall%binary_conv2d/StatefulPartitionedCall2:
bn1/StatefulPartitionedCallbn1/StatefulPartitionedCall2:
bn5/StatefulPartitionedCallbn5/StatefulPartitionedCall2:
bn6/StatefulPartitionedCallbn6/StatefulPartitionedCall2>
conv1/StatefulPartitionedCallconv1/StatefulPartitionedCall2@
dense5/StatefulPartitionedCalldense5/StatefulPartitionedCall2@
dense6/StatefulPartitionedCalldense6/StatefulPartitionedCall:\ X
/
_output_shapes
:?????????@@
%
_user_specified_nameconv1_input
?
?
*__inference_sequential_layer_call_fn_27360

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18* 
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d*6
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_267202
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????d2

Identity"
identityIdentity:output:0*~
_input_shapesm
k:?????????@@::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????@@
 
_user_specified_nameinputs
?
?
>__inference_bn1_layer_call_and_return_conditional_losses_27444

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????   : : : : :*
data_formatNCHW*
epsilon%??'7*
exponential_avg_factor%???=2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:?????????   2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????   ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????   
 
_user_specified_nameinputs
?
?
3__inference_batch_normalization_layer_call_fn_27641

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_258022
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?i
?
__inference__traced_save_28116
file_prefix+
'savev2_conv1_kernel_read_readvariableop(
$savev2_bn1_gamma_read_readvariableop'
#savev2_bn1_beta_read_readvariableop.
*savev2_bn1_moving_mean_read_readvariableop2
.savev2_bn1_moving_variance_read_readvariableop3
/savev2_binary_conv2d_kernel_read_readvariableop8
4savev2_batch_normalization_gamma_read_readvariableop7
3savev2_batch_normalization_beta_read_readvariableop>
:savev2_batch_normalization_moving_mean_read_readvariableopB
>savev2_batch_normalization_moving_variance_read_readvariableop,
(savev2_dense5_kernel_read_readvariableop(
$savev2_bn5_gamma_read_readvariableop'
#savev2_bn5_beta_read_readvariableop.
*savev2_bn5_moving_mean_read_readvariableop2
.savev2_bn5_moving_variance_read_readvariableop,
(savev2_dense6_kernel_read_readvariableop(
$savev2_bn6_gamma_read_readvariableop'
#savev2_bn6_beta_read_readvariableop.
*savev2_bn6_moving_mean_read_readvariableop2
.savev2_bn6_moving_variance_read_readvariableop%
!savev2_beta_1_read_readvariableop%
!savev2_beta_2_read_readvariableop$
 savev2_decay_read_readvariableop,
(savev2_learning_rate_read_readvariableop(
$savev2_adam_iter_read_readvariableop	$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop2
.savev2_adam_conv1_kernel_m_read_readvariableop/
+savev2_adam_bn1_gamma_m_read_readvariableop.
*savev2_adam_bn1_beta_m_read_readvariableop:
6savev2_adam_binary_conv2d_kernel_m_read_readvariableop?
;savev2_adam_batch_normalization_gamma_m_read_readvariableop>
:savev2_adam_batch_normalization_beta_m_read_readvariableop3
/savev2_adam_dense5_kernel_m_read_readvariableop/
+savev2_adam_bn5_gamma_m_read_readvariableop.
*savev2_adam_bn5_beta_m_read_readvariableop3
/savev2_adam_dense6_kernel_m_read_readvariableop/
+savev2_adam_bn6_gamma_m_read_readvariableop.
*savev2_adam_bn6_beta_m_read_readvariableop2
.savev2_adam_conv1_kernel_v_read_readvariableop/
+savev2_adam_bn1_gamma_v_read_readvariableop.
*savev2_adam_bn1_beta_v_read_readvariableop:
6savev2_adam_binary_conv2d_kernel_v_read_readvariableop?
;savev2_adam_batch_normalization_gamma_v_read_readvariableop>
:savev2_adam_batch_normalization_beta_v_read_readvariableop3
/savev2_adam_dense5_kernel_v_read_readvariableop/
+savev2_adam_bn5_gamma_v_read_readvariableop.
*savev2_adam_bn5_beta_v_read_readvariableop3
/savev2_adam_dense6_kernel_v_read_readvariableop/
+savev2_adam_bn6_gamma_v_read_readvariableop.
*savev2_adam_bn6_beta_v_read_readvariableop
savev2_const

identity_1??MergeV2Checkpoints?
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:6*
dtype0*?
value?B?6B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:6*
dtype0*
valuevBt6B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0'savev2_conv1_kernel_read_readvariableop$savev2_bn1_gamma_read_readvariableop#savev2_bn1_beta_read_readvariableop*savev2_bn1_moving_mean_read_readvariableop.savev2_bn1_moving_variance_read_readvariableop/savev2_binary_conv2d_kernel_read_readvariableop4savev2_batch_normalization_gamma_read_readvariableop3savev2_batch_normalization_beta_read_readvariableop:savev2_batch_normalization_moving_mean_read_readvariableop>savev2_batch_normalization_moving_variance_read_readvariableop(savev2_dense5_kernel_read_readvariableop$savev2_bn5_gamma_read_readvariableop#savev2_bn5_beta_read_readvariableop*savev2_bn5_moving_mean_read_readvariableop.savev2_bn5_moving_variance_read_readvariableop(savev2_dense6_kernel_read_readvariableop$savev2_bn6_gamma_read_readvariableop#savev2_bn6_beta_read_readvariableop*savev2_bn6_moving_mean_read_readvariableop.savev2_bn6_moving_variance_read_readvariableop!savev2_beta_1_read_readvariableop!savev2_beta_2_read_readvariableop savev2_decay_read_readvariableop(savev2_learning_rate_read_readvariableop$savev2_adam_iter_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop.savev2_adam_conv1_kernel_m_read_readvariableop+savev2_adam_bn1_gamma_m_read_readvariableop*savev2_adam_bn1_beta_m_read_readvariableop6savev2_adam_binary_conv2d_kernel_m_read_readvariableop;savev2_adam_batch_normalization_gamma_m_read_readvariableop:savev2_adam_batch_normalization_beta_m_read_readvariableop/savev2_adam_dense5_kernel_m_read_readvariableop+savev2_adam_bn5_gamma_m_read_readvariableop*savev2_adam_bn5_beta_m_read_readvariableop/savev2_adam_dense6_kernel_m_read_readvariableop+savev2_adam_bn6_gamma_m_read_readvariableop*savev2_adam_bn6_beta_m_read_readvariableop.savev2_adam_conv1_kernel_v_read_readvariableop+savev2_adam_bn1_gamma_v_read_readvariableop*savev2_adam_bn1_beta_v_read_readvariableop6savev2_adam_binary_conv2d_kernel_v_read_readvariableop;savev2_adam_batch_normalization_gamma_v_read_readvariableop:savev2_adam_batch_normalization_beta_v_read_readvariableop/savev2_adam_dense5_kernel_v_read_readvariableop+savev2_adam_bn5_gamma_v_read_readvariableop*savev2_adam_bn5_beta_v_read_readvariableop/savev2_adam_dense6_kernel_v_read_readvariableop+savev2_adam_bn6_gamma_v_read_readvariableop*savev2_adam_bn6_beta_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *D
dtypes:
826	2
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*?
_input_shapes?
?: : : : : : :  :::::
?@?:?:?:?:?:	?d:d:d:d:d: : : : : : : : : : : : :  :::
?@?:?:?:	?d:d:d: : : :  :::
?@?:?:?:	?d:d:d: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
:  : 

_output_shapes
:: 

_output_shapes
:: 	

_output_shapes
:: 


_output_shapes
::&"
 
_output_shapes
:
?@?:!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:%!

_output_shapes
:	?d: 

_output_shapes
:d: 

_output_shapes
:d: 

_output_shapes
:d: 

_output_shapes
:d:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :,(
&
_output_shapes
: : 

_output_shapes
: :  

_output_shapes
: :,!(
&
_output_shapes
:  : "

_output_shapes
:: #

_output_shapes
::&$"
 
_output_shapes
:
?@?:!%

_output_shapes	
:?:!&

_output_shapes	
:?:%'!

_output_shapes
:	?d: (

_output_shapes
:d: )

_output_shapes
:d:,*(
&
_output_shapes
: : +

_output_shapes
: : ,

_output_shapes
: :,-(
&
_output_shapes
:  : .

_output_shapes
:: /

_output_shapes
::&0"
 
_output_shapes
:
?@?:!1

_output_shapes	
:?:!2

_output_shapes	
:?:%3!

_output_shapes
:	?d: 4

_output_shapes
:d: 5

_output_shapes
:d:6

_output_shapes
: 
?
?
@__inference_conv1_layer_call_and_return_conditional_losses_25537

inputs
readvariableop_resource
identity??ReadVariableOp?
ReadVariableOpReadVariableOpreadvariableop_resource*&
_output_shapes
: *
dtype02
ReadVariableOp[
	truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
	truediv/yz
truedivRealDivReadVariableOp:value:0truediv/y:output:0*
T0*&
_output_shapes
: 2	
truedivS
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
mul/x_
mulMulmul/x:output:0truediv:z:0*
T0*&
_output_shapes
: 2
mulS
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
add/y]
addAddV2mul:z:0add/y:output:0*
T0*&
_output_shapes
: 2
addw
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value/Minimum/y?
clip_by_value/MinimumMinimumadd:z:0 clip_by_value/Minimum/y:output:0*
T0*&
_output_shapes
: 2
clip_by_value/Minimumg
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value/y?
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*&
_output_shapes
: 2
clip_by_value[
RoundRoundclip_by_value:z:0*
T0*&
_output_shapes
: 2
Round`
subSub	Round:y:0clip_by_value:z:0*
T0*&
_output_shapes
: 2
subf
StopGradientStopGradientsub:z:0*
T0*&
_output_shapes
: 2
StopGradientr
add_1AddV2clip_by_value:z:0StopGradient:output:0*
T0*&
_output_shapes
: 2
add_1W
mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @2	
mul_1/xc
mul_1Mulmul_1/x:output:0	add_1:z:0*
T0*&
_output_shapes
: 2
mul_1W
sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2	
sub_1/yc
sub_1Sub	mul_1:z:0sub_1/y:output:0*
T0*&
_output_shapes
: 2
sub_1W
mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2	
mul_2/xc
mul_2Mulmul_2/x:output:0	sub_1:z:0*
T0*&
_output_shapes
: 2
mul_2?
convolutionConv2Dinputs	mul_2:z:0*
T0*A
_output_shapes/
-:+??????????????????????????? *
paddingSAME*
strides
2
convolution?
IdentityIdentityconvolution:output:0^ReadVariableOp*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*D
_input_shapes3
1:+???????????????????????????:2 
ReadVariableOpReadVariableOp:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
??
?
E__inference_sequential_layer_call_and_return_conditional_losses_27062

inputs!
conv1_readvariableop_resource
bn1_readvariableop_resource!
bn1_readvariableop_1_resource0
,bn1_fusedbatchnormv3_readvariableop_resource2
.bn1_fusedbatchnormv3_readvariableop_1_resource)
%binary_conv2d_readvariableop_resource/
+batch_normalization_readvariableop_resource1
-batch_normalization_readvariableop_1_resource@
<batch_normalization_fusedbatchnormv3_readvariableop_resourceB
>batch_normalization_fusedbatchnormv3_readvariableop_1_resource"
dense5_readvariableop_resource
bn5_assignmovingavg_26966
bn5_assignmovingavg_1_26972-
)bn5_batchnorm_mul_readvariableop_resource)
%bn5_batchnorm_readvariableop_resource"
dense6_readvariableop_resource
bn6_assignmovingavg_27037
bn6_assignmovingavg_1_27043-
)bn6_batchnorm_mul_readvariableop_resource)
%bn6_batchnorm_readvariableop_resource
identity??"batch_normalization/AssignNewValue?$batch_normalization/AssignNewValue_1?3batch_normalization/FusedBatchNormV3/ReadVariableOp?5batch_normalization/FusedBatchNormV3/ReadVariableOp_1?"batch_normalization/ReadVariableOp?$batch_normalization/ReadVariableOp_1?binary_conv2d/ReadVariableOp?bn1/AssignNewValue?bn1/AssignNewValue_1?#bn1/FusedBatchNormV3/ReadVariableOp?%bn1/FusedBatchNormV3/ReadVariableOp_1?bn1/ReadVariableOp?bn1/ReadVariableOp_1?'bn5/AssignMovingAvg/AssignSubVariableOp?"bn5/AssignMovingAvg/ReadVariableOp?)bn5/AssignMovingAvg_1/AssignSubVariableOp?$bn5/AssignMovingAvg_1/ReadVariableOp?bn5/batchnorm/ReadVariableOp? bn5/batchnorm/mul/ReadVariableOp?'bn6/AssignMovingAvg/AssignSubVariableOp?"bn6/AssignMovingAvg/ReadVariableOp?)bn6/AssignMovingAvg_1/AssignSubVariableOp?$bn6/AssignMovingAvg_1/ReadVariableOp?bn6/batchnorm/ReadVariableOp? bn6/batchnorm/mul/ReadVariableOp?conv1/ReadVariableOp?dense5/ReadVariableOp?dense6/ReadVariableOp?
conv1/ReadVariableOpReadVariableOpconv1_readvariableop_resource*&
_output_shapes
: *
dtype02
conv1/ReadVariableOpg
conv1/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
conv1/truediv/y?
conv1/truedivRealDivconv1/ReadVariableOp:value:0conv1/truediv/y:output:0*
T0*&
_output_shapes
: 2
conv1/truediv_
conv1/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
conv1/mul/xw
	conv1/mulMulconv1/mul/x:output:0conv1/truediv:z:0*
T0*&
_output_shapes
: 2
	conv1/mul_
conv1/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
conv1/add/yu
	conv1/addAddV2conv1/mul:z:0conv1/add/y:output:0*
T0*&
_output_shapes
: 2
	conv1/add?
conv1/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
conv1/clip_by_value/Minimum/y?
conv1/clip_by_value/MinimumMinimumconv1/add:z:0&conv1/clip_by_value/Minimum/y:output:0*
T0*&
_output_shapes
: 2
conv1/clip_by_value/Minimums
conv1/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
conv1/clip_by_value/y?
conv1/clip_by_valueMaximumconv1/clip_by_value/Minimum:z:0conv1/clip_by_value/y:output:0*
T0*&
_output_shapes
: 2
conv1/clip_by_valuem
conv1/RoundRoundconv1/clip_by_value:z:0*
T0*&
_output_shapes
: 2
conv1/Roundx
	conv1/subSubconv1/Round:y:0conv1/clip_by_value:z:0*
T0*&
_output_shapes
: 2
	conv1/subx
conv1/StopGradientStopGradientconv1/sub:z:0*
T0*&
_output_shapes
: 2
conv1/StopGradient?
conv1/add_1AddV2conv1/clip_by_value:z:0conv1/StopGradient:output:0*
T0*&
_output_shapes
: 2
conv1/add_1c
conv1/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
conv1/mul_1/x{
conv1/mul_1Mulconv1/mul_1/x:output:0conv1/add_1:z:0*
T0*&
_output_shapes
: 2
conv1/mul_1c
conv1/sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
conv1/sub_1/y{
conv1/sub_1Subconv1/mul_1:z:0conv1/sub_1/y:output:0*
T0*&
_output_shapes
: 2
conv1/sub_1c
conv1/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
conv1/mul_2/x{
conv1/mul_2Mulconv1/mul_2/x:output:0conv1/sub_1:z:0*
T0*&
_output_shapes
: 2
conv1/mul_2?
conv1/convolutionConv2Dinputsconv1/mul_2:z:0*
T0*/
_output_shapes
:?????????@@ *
paddingSAME*
strides
2
conv1/convolution?
max_pooling2d/MaxPoolMaxPoolconv1/convolution:output:0*/
_output_shapes
:?????????   *
ksize
*
paddingVALID*
strides
2
max_pooling2d/MaxPool?
bn1/ReadVariableOpReadVariableOpbn1_readvariableop_resource*
_output_shapes
: *
dtype02
bn1/ReadVariableOp?
bn1/ReadVariableOp_1ReadVariableOpbn1_readvariableop_1_resource*
_output_shapes
: *
dtype02
bn1/ReadVariableOp_1?
#bn1/FusedBatchNormV3/ReadVariableOpReadVariableOp,bn1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02%
#bn1/FusedBatchNormV3/ReadVariableOp?
%bn1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp.bn1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02'
%bn1/FusedBatchNormV3/ReadVariableOp_1?
bn1/FusedBatchNormV3FusedBatchNormV3max_pooling2d/MaxPool:output:0bn1/ReadVariableOp:value:0bn1/ReadVariableOp_1:value:0+bn1/FusedBatchNormV3/ReadVariableOp:value:0-bn1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????   : : : : :*
data_formatNCHW*
epsilon%??'7*
exponential_avg_factor%???=2
bn1/FusedBatchNormV3?
bn1/AssignNewValueAssignVariableOp,bn1_fusedbatchnormv3_readvariableop_resource!bn1/FusedBatchNormV3:batch_mean:0$^bn1/FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*?
_class5
31loc:@bn1/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
bn1/AssignNewValue?
bn1/AssignNewValue_1AssignVariableOp.bn1_fusedbatchnormv3_readvariableop_1_resource%bn1/FusedBatchNormV3:batch_variance:0&^bn1/FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*A
_class7
53loc:@bn1/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
bn1/AssignNewValue_1]

act1/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2

act1/mul/x?
act1/mulMulact1/mul/x:output:0bn1/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????   2

act1/mul]

act1/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2

act1/add/yz
act1/addAddV2act1/mul:z:0act1/add/y:output:0*
T0*/
_output_shapes
:?????????   2

act1/add?
act1/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
act1/clip_by_value/Minimum/y?
act1/clip_by_value/MinimumMinimumact1/add:z:0%act1/clip_by_value/Minimum/y:output:0*
T0*/
_output_shapes
:?????????   2
act1/clip_by_value/Minimumq
act1/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
act1/clip_by_value/y?
act1/clip_by_valueMaximumact1/clip_by_value/Minimum:z:0act1/clip_by_value/y:output:0*
T0*/
_output_shapes
:?????????   2
act1/clip_by_values

act1/RoundRoundact1/clip_by_value:z:0*
T0*/
_output_shapes
:?????????   2

act1/Round}
act1/subSubact1/Round:y:0act1/clip_by_value:z:0*
T0*/
_output_shapes
:?????????   2

act1/sub~
act1/StopGradientStopGradientact1/sub:z:0*
T0*/
_output_shapes
:?????????   2
act1/StopGradient?

act1/add_1AddV2act1/clip_by_value:z:0act1/StopGradient:output:0*
T0*/
_output_shapes
:?????????   2

act1/add_1a
act1/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
act1/mul_1/x?

act1/mul_1Mulact1/mul_1/x:output:0act1/add_1:z:0*
T0*/
_output_shapes
:?????????   2

act1/mul_1a
act1/sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
act1/sub_1/y?

act1/sub_1Subact1/mul_1:z:0act1/sub_1/y:output:0*
T0*/
_output_shapes
:?????????   2

act1/sub_1?
binary_conv2d/ReadVariableOpReadVariableOp%binary_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02
binary_conv2d/ReadVariableOpw
binary_conv2d/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
binary_conv2d/truediv/y?
binary_conv2d/truedivRealDiv$binary_conv2d/ReadVariableOp:value:0 binary_conv2d/truediv/y:output:0*
T0*&
_output_shapes
:  2
binary_conv2d/truedivo
binary_conv2d/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
binary_conv2d/mul/x?
binary_conv2d/mulMulbinary_conv2d/mul/x:output:0binary_conv2d/truediv:z:0*
T0*&
_output_shapes
:  2
binary_conv2d/mulo
binary_conv2d/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
binary_conv2d/add/y?
binary_conv2d/addAddV2binary_conv2d/mul:z:0binary_conv2d/add/y:output:0*
T0*&
_output_shapes
:  2
binary_conv2d/add?
%binary_conv2d/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2'
%binary_conv2d/clip_by_value/Minimum/y?
#binary_conv2d/clip_by_value/MinimumMinimumbinary_conv2d/add:z:0.binary_conv2d/clip_by_value/Minimum/y:output:0*
T0*&
_output_shapes
:  2%
#binary_conv2d/clip_by_value/Minimum?
binary_conv2d/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
binary_conv2d/clip_by_value/y?
binary_conv2d/clip_by_valueMaximum'binary_conv2d/clip_by_value/Minimum:z:0&binary_conv2d/clip_by_value/y:output:0*
T0*&
_output_shapes
:  2
binary_conv2d/clip_by_value?
binary_conv2d/RoundRoundbinary_conv2d/clip_by_value:z:0*
T0*&
_output_shapes
:  2
binary_conv2d/Round?
binary_conv2d/subSubbinary_conv2d/Round:y:0binary_conv2d/clip_by_value:z:0*
T0*&
_output_shapes
:  2
binary_conv2d/sub?
binary_conv2d/StopGradientStopGradientbinary_conv2d/sub:z:0*
T0*&
_output_shapes
:  2
binary_conv2d/StopGradient?
binary_conv2d/add_1AddV2binary_conv2d/clip_by_value:z:0#binary_conv2d/StopGradient:output:0*
T0*&
_output_shapes
:  2
binary_conv2d/add_1s
binary_conv2d/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
binary_conv2d/mul_1/x?
binary_conv2d/mul_1Mulbinary_conv2d/mul_1/x:output:0binary_conv2d/add_1:z:0*
T0*&
_output_shapes
:  2
binary_conv2d/mul_1s
binary_conv2d/sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
binary_conv2d/sub_1/y?
binary_conv2d/sub_1Subbinary_conv2d/mul_1:z:0binary_conv2d/sub_1/y:output:0*
T0*&
_output_shapes
:  2
binary_conv2d/sub_1s
binary_conv2d/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
binary_conv2d/mul_2/x?
binary_conv2d/mul_2Mulbinary_conv2d/mul_2/x:output:0binary_conv2d/sub_1:z:0*
T0*&
_output_shapes
:  2
binary_conv2d/mul_2?
binary_conv2d/convolutionConv2Dact1/sub_1:z:0binary_conv2d/mul_2:z:0*
T0*/
_output_shapes
:?????????   *
paddingSAME*
strides
2
binary_conv2d/convolution?
max_pooling2d_1/MaxPoolMaxPool"binary_conv2d/convolution:output:0*/
_output_shapes
:????????? *
ksize
*
paddingVALID*
strides
2
max_pooling2d_1/MaxPool?
"batch_normalization/ReadVariableOpReadVariableOp+batch_normalization_readvariableop_resource*
_output_shapes
:*
dtype02$
"batch_normalization/ReadVariableOp?
$batch_normalization/ReadVariableOp_1ReadVariableOp-batch_normalization_readvariableop_1_resource*
_output_shapes
:*
dtype02&
$batch_normalization/ReadVariableOp_1?
3batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype025
3batch_normalization/FusedBatchNormV3/ReadVariableOp?
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype027
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1?
$batch_normalization/FusedBatchNormV3FusedBatchNormV3 max_pooling2d_1/MaxPool:output:0*batch_normalization/ReadVariableOp:value:0,batch_normalization/ReadVariableOp_1:value:0;batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0=batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:????????? :::::*
data_formatNCHW*
epsilon%??'7*
exponential_avg_factor%???=2&
$batch_normalization/FusedBatchNormV3?
"batch_normalization/AssignNewValueAssignVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource1batch_normalization/FusedBatchNormV3:batch_mean:04^batch_normalization/FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*O
_classE
CAloc:@batch_normalization/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02$
"batch_normalization/AssignNewValue?
$batch_normalization/AssignNewValue_1AssignVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource5batch_normalization/FusedBatchNormV3:batch_variance:06^batch_normalization/FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*Q
_classG
ECloc:@batch_normalization/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02&
$batch_normalization/AssignNewValue_1i
activation/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
activation/mul/x?
activation/mulMulactivation/mul/x:output:0(batch_normalization/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:????????? 2
activation/muli
activation/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
activation/add/y?
activation/addAddV2activation/mul:z:0activation/add/y:output:0*
T0*/
_output_shapes
:????????? 2
activation/add?
"activation/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2$
"activation/clip_by_value/Minimum/y?
 activation/clip_by_value/MinimumMinimumactivation/add:z:0+activation/clip_by_value/Minimum/y:output:0*
T0*/
_output_shapes
:????????? 2"
 activation/clip_by_value/Minimum}
activation/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
activation/clip_by_value/y?
activation/clip_by_valueMaximum$activation/clip_by_value/Minimum:z:0#activation/clip_by_value/y:output:0*
T0*/
_output_shapes
:????????? 2
activation/clip_by_value?
activation/RoundRoundactivation/clip_by_value:z:0*
T0*/
_output_shapes
:????????? 2
activation/Round?
activation/subSubactivation/Round:y:0activation/clip_by_value:z:0*
T0*/
_output_shapes
:????????? 2
activation/sub?
activation/StopGradientStopGradientactivation/sub:z:0*
T0*/
_output_shapes
:????????? 2
activation/StopGradient?
activation/add_1AddV2activation/clip_by_value:z:0 activation/StopGradient:output:0*
T0*/
_output_shapes
:????????? 2
activation/add_1m
activation/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
activation/mul_1/x?
activation/mul_1Mulactivation/mul_1/x:output:0activation/add_1:z:0*
T0*/
_output_shapes
:????????? 2
activation/mul_1m
activation/sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
activation/sub_1/y?
activation/sub_1Subactivation/mul_1:z:0activation/sub_1/y:output:0*
T0*/
_output_shapes
:????????? 2
activation/sub_1o
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????    2
flatten/Const?
flatten/ReshapeReshapeactivation/sub_1:z:0flatten/Const:output:0*
T0*(
_output_shapes
:??????????@2
flatten/Reshape?
dense5/ReadVariableOpReadVariableOpdense5_readvariableop_resource* 
_output_shapes
:
?@?*
dtype02
dense5/ReadVariableOpi
dense5/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dense5/truediv/y?
dense5/truedivRealDivdense5/ReadVariableOp:value:0dense5/truediv/y:output:0*
T0* 
_output_shapes
:
?@?2
dense5/truediva
dense5/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dense5/mul/xu

dense5/mulMuldense5/mul/x:output:0dense5/truediv:z:0*
T0* 
_output_shapes
:
?@?2

dense5/mula
dense5/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dense5/add/ys

dense5/addAddV2dense5/mul:z:0dense5/add/y:output:0*
T0* 
_output_shapes
:
?@?2

dense5/add?
dense5/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2 
dense5/clip_by_value/Minimum/y?
dense5/clip_by_value/MinimumMinimumdense5/add:z:0'dense5/clip_by_value/Minimum/y:output:0*
T0* 
_output_shapes
:
?@?2
dense5/clip_by_value/Minimumu
dense5/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
dense5/clip_by_value/y?
dense5/clip_by_valueMaximum dense5/clip_by_value/Minimum:z:0dense5/clip_by_value/y:output:0*
T0* 
_output_shapes
:
?@?2
dense5/clip_by_valuej
dense5/RoundRounddense5/clip_by_value:z:0*
T0* 
_output_shapes
:
?@?2
dense5/Roundv

dense5/subSubdense5/Round:y:0dense5/clip_by_value:z:0*
T0* 
_output_shapes
:
?@?2

dense5/subu
dense5/StopGradientStopGradientdense5/sub:z:0*
T0* 
_output_shapes
:
?@?2
dense5/StopGradient?
dense5/add_1AddV2dense5/clip_by_value:z:0dense5/StopGradient:output:0*
T0* 
_output_shapes
:
?@?2
dense5/add_1e
dense5/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dense5/mul_1/xy
dense5/mul_1Muldense5/mul_1/x:output:0dense5/add_1:z:0*
T0* 
_output_shapes
:
?@?2
dense5/mul_1e
dense5/sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dense5/sub_1/yy
dense5/sub_1Subdense5/mul_1:z:0dense5/sub_1/y:output:0*
T0* 
_output_shapes
:
?@?2
dense5/sub_1e
dense5/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dense5/mul_2/xy
dense5/mul_2Muldense5/mul_2/x:output:0dense5/sub_1:z:0*
T0* 
_output_shapes
:
?@?2
dense5/mul_2?
dense5/MatMulMatMulflatten/Reshape:output:0dense5/mul_2:z:0*
T0*(
_output_shapes
:??????????2
dense5/MatMul?
"bn5/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"bn5/moments/mean/reduction_indices?
bn5/moments/meanMeandense5/MatMul:product:0+bn5/moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	?*
	keep_dims(2
bn5/moments/mean?
bn5/moments/StopGradientStopGradientbn5/moments/mean:output:0*
T0*
_output_shapes
:	?2
bn5/moments/StopGradient?
bn5/moments/SquaredDifferenceSquaredDifferencedense5/MatMul:product:0!bn5/moments/StopGradient:output:0*
T0*(
_output_shapes
:??????????2
bn5/moments/SquaredDifference?
&bn5/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2(
&bn5/moments/variance/reduction_indices?
bn5/moments/varianceMean!bn5/moments/SquaredDifference:z:0/bn5/moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	?*
	keep_dims(2
bn5/moments/variance?
bn5/moments/SqueezeSqueezebn5/moments/mean:output:0*
T0*
_output_shapes	
:?*
squeeze_dims
 2
bn5/moments/Squeeze?
bn5/moments/Squeeze_1Squeezebn5/moments/variance:output:0*
T0*
_output_shapes	
:?*
squeeze_dims
 2
bn5/moments/Squeeze_1?
bn5/AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*,
_class"
 loc:@bn5/AssignMovingAvg/26966*
_output_shapes
: *
dtype0*
valueB
 *???=2
bn5/AssignMovingAvg/decay?
"bn5/AssignMovingAvg/ReadVariableOpReadVariableOpbn5_assignmovingavg_26966*
_output_shapes	
:?*
dtype02$
"bn5/AssignMovingAvg/ReadVariableOp?
bn5/AssignMovingAvg/subSub*bn5/AssignMovingAvg/ReadVariableOp:value:0bn5/moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*,
_class"
 loc:@bn5/AssignMovingAvg/26966*
_output_shapes	
:?2
bn5/AssignMovingAvg/sub?
bn5/AssignMovingAvg/mulMulbn5/AssignMovingAvg/sub:z:0"bn5/AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*,
_class"
 loc:@bn5/AssignMovingAvg/26966*
_output_shapes	
:?2
bn5/AssignMovingAvg/mul?
'bn5/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpbn5_assignmovingavg_26966bn5/AssignMovingAvg/mul:z:0#^bn5/AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*,
_class"
 loc:@bn5/AssignMovingAvg/26966*
_output_shapes
 *
dtype02)
'bn5/AssignMovingAvg/AssignSubVariableOp?
bn5/AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*.
_class$
" loc:@bn5/AssignMovingAvg_1/26972*
_output_shapes
: *
dtype0*
valueB
 *???=2
bn5/AssignMovingAvg_1/decay?
$bn5/AssignMovingAvg_1/ReadVariableOpReadVariableOpbn5_assignmovingavg_1_26972*
_output_shapes	
:?*
dtype02&
$bn5/AssignMovingAvg_1/ReadVariableOp?
bn5/AssignMovingAvg_1/subSub,bn5/AssignMovingAvg_1/ReadVariableOp:value:0bn5/moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*.
_class$
" loc:@bn5/AssignMovingAvg_1/26972*
_output_shapes	
:?2
bn5/AssignMovingAvg_1/sub?
bn5/AssignMovingAvg_1/mulMulbn5/AssignMovingAvg_1/sub:z:0$bn5/AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*.
_class$
" loc:@bn5/AssignMovingAvg_1/26972*
_output_shapes	
:?2
bn5/AssignMovingAvg_1/mul?
)bn5/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpbn5_assignmovingavg_1_26972bn5/AssignMovingAvg_1/mul:z:0%^bn5/AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*.
_class$
" loc:@bn5/AssignMovingAvg_1/26972*
_output_shapes
 *
dtype02+
)bn5/AssignMovingAvg_1/AssignSubVariableOpo
bn5/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *?7?52
bn5/batchnorm/add/y?
bn5/batchnorm/addAddV2bn5/moments/Squeeze_1:output:0bn5/batchnorm/add/y:output:0*
T0*
_output_shapes	
:?2
bn5/batchnorm/addp
bn5/batchnorm/RsqrtRsqrtbn5/batchnorm/add:z:0*
T0*
_output_shapes	
:?2
bn5/batchnorm/Rsqrt?
 bn5/batchnorm/mul/ReadVariableOpReadVariableOp)bn5_batchnorm_mul_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 bn5/batchnorm/mul/ReadVariableOp?
bn5/batchnorm/mulMulbn5/batchnorm/Rsqrt:y:0(bn5/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:?2
bn5/batchnorm/mul?
bn5/batchnorm/mul_1Muldense5/MatMul:product:0bn5/batchnorm/mul:z:0*
T0*(
_output_shapes
:??????????2
bn5/batchnorm/mul_1?
bn5/batchnorm/mul_2Mulbn5/moments/Squeeze:output:0bn5/batchnorm/mul:z:0*
T0*
_output_shapes	
:?2
bn5/batchnorm/mul_2?
bn5/batchnorm/ReadVariableOpReadVariableOp%bn5_batchnorm_readvariableop_resource*
_output_shapes	
:?*
dtype02
bn5/batchnorm/ReadVariableOp?
bn5/batchnorm/subSub$bn5/batchnorm/ReadVariableOp:value:0bn5/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?2
bn5/batchnorm/sub?
bn5/batchnorm/add_1AddV2bn5/batchnorm/mul_1:z:0bn5/batchnorm/sub:z:0*
T0*(
_output_shapes
:??????????2
bn5/batchnorm/add_1]

act5/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2

act5/mul/x|
act5/mulMulact5/mul/x:output:0bn5/batchnorm/add_1:z:0*
T0*(
_output_shapes
:??????????2

act5/mul]

act5/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2

act5/add/ys
act5/addAddV2act5/mul:z:0act5/add/y:output:0*
T0*(
_output_shapes
:??????????2

act5/add?
act5/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
act5/clip_by_value/Minimum/y?
act5/clip_by_value/MinimumMinimumact5/add:z:0%act5/clip_by_value/Minimum/y:output:0*
T0*(
_output_shapes
:??????????2
act5/clip_by_value/Minimumq
act5/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
act5/clip_by_value/y?
act5/clip_by_valueMaximumact5/clip_by_value/Minimum:z:0act5/clip_by_value/y:output:0*
T0*(
_output_shapes
:??????????2
act5/clip_by_valuel

act5/RoundRoundact5/clip_by_value:z:0*
T0*(
_output_shapes
:??????????2

act5/Roundv
act5/subSubact5/Round:y:0act5/clip_by_value:z:0*
T0*(
_output_shapes
:??????????2

act5/subw
act5/StopGradientStopGradientact5/sub:z:0*
T0*(
_output_shapes
:??????????2
act5/StopGradient?

act5/add_1AddV2act5/clip_by_value:z:0act5/StopGradient:output:0*
T0*(
_output_shapes
:??????????2

act5/add_1a
act5/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
act5/mul_1/xy

act5/mul_1Mulact5/mul_1/x:output:0act5/add_1:z:0*
T0*(
_output_shapes
:??????????2

act5/mul_1a
act5/sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
act5/sub_1/yy

act5/sub_1Subact5/mul_1:z:0act5/sub_1/y:output:0*
T0*(
_output_shapes
:??????????2

act5/sub_1?
dense6/ReadVariableOpReadVariableOpdense6_readvariableop_resource*
_output_shapes
:	?d*
dtype02
dense6/ReadVariableOpi
dense6/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dense6/truediv/y?
dense6/truedivRealDivdense6/ReadVariableOp:value:0dense6/truediv/y:output:0*
T0*
_output_shapes
:	?d2
dense6/truediva
dense6/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dense6/mul/xt

dense6/mulMuldense6/mul/x:output:0dense6/truediv:z:0*
T0*
_output_shapes
:	?d2

dense6/mula
dense6/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dense6/add/yr

dense6/addAddV2dense6/mul:z:0dense6/add/y:output:0*
T0*
_output_shapes
:	?d2

dense6/add?
dense6/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2 
dense6/clip_by_value/Minimum/y?
dense6/clip_by_value/MinimumMinimumdense6/add:z:0'dense6/clip_by_value/Minimum/y:output:0*
T0*
_output_shapes
:	?d2
dense6/clip_by_value/Minimumu
dense6/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
dense6/clip_by_value/y?
dense6/clip_by_valueMaximum dense6/clip_by_value/Minimum:z:0dense6/clip_by_value/y:output:0*
T0*
_output_shapes
:	?d2
dense6/clip_by_valuei
dense6/RoundRounddense6/clip_by_value:z:0*
T0*
_output_shapes
:	?d2
dense6/Roundu

dense6/subSubdense6/Round:y:0dense6/clip_by_value:z:0*
T0*
_output_shapes
:	?d2

dense6/subt
dense6/StopGradientStopGradientdense6/sub:z:0*
T0*
_output_shapes
:	?d2
dense6/StopGradient?
dense6/add_1AddV2dense6/clip_by_value:z:0dense6/StopGradient:output:0*
T0*
_output_shapes
:	?d2
dense6/add_1e
dense6/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dense6/mul_1/xx
dense6/mul_1Muldense6/mul_1/x:output:0dense6/add_1:z:0*
T0*
_output_shapes
:	?d2
dense6/mul_1e
dense6/sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dense6/sub_1/yx
dense6/sub_1Subdense6/mul_1:z:0dense6/sub_1/y:output:0*
T0*
_output_shapes
:	?d2
dense6/sub_1e
dense6/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dense6/mul_2/xx
dense6/mul_2Muldense6/mul_2/x:output:0dense6/sub_1:z:0*
T0*
_output_shapes
:	?d2
dense6/mul_2|
dense6/MatMulMatMulact5/sub_1:z:0dense6/mul_2:z:0*
T0*'
_output_shapes
:?????????d2
dense6/MatMul?
"bn6/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"bn6/moments/mean/reduction_indices?
bn6/moments/meanMeandense6/MatMul:product:0+bn6/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:d*
	keep_dims(2
bn6/moments/mean?
bn6/moments/StopGradientStopGradientbn6/moments/mean:output:0*
T0*
_output_shapes

:d2
bn6/moments/StopGradient?
bn6/moments/SquaredDifferenceSquaredDifferencedense6/MatMul:product:0!bn6/moments/StopGradient:output:0*
T0*'
_output_shapes
:?????????d2
bn6/moments/SquaredDifference?
&bn6/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2(
&bn6/moments/variance/reduction_indices?
bn6/moments/varianceMean!bn6/moments/SquaredDifference:z:0/bn6/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:d*
	keep_dims(2
bn6/moments/variance?
bn6/moments/SqueezeSqueezebn6/moments/mean:output:0*
T0*
_output_shapes
:d*
squeeze_dims
 2
bn6/moments/Squeeze?
bn6/moments/Squeeze_1Squeezebn6/moments/variance:output:0*
T0*
_output_shapes
:d*
squeeze_dims
 2
bn6/moments/Squeeze_1?
bn6/AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*,
_class"
 loc:@bn6/AssignMovingAvg/27037*
_output_shapes
: *
dtype0*
valueB
 *???=2
bn6/AssignMovingAvg/decay?
"bn6/AssignMovingAvg/ReadVariableOpReadVariableOpbn6_assignmovingavg_27037*
_output_shapes
:d*
dtype02$
"bn6/AssignMovingAvg/ReadVariableOp?
bn6/AssignMovingAvg/subSub*bn6/AssignMovingAvg/ReadVariableOp:value:0bn6/moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*,
_class"
 loc:@bn6/AssignMovingAvg/27037*
_output_shapes
:d2
bn6/AssignMovingAvg/sub?
bn6/AssignMovingAvg/mulMulbn6/AssignMovingAvg/sub:z:0"bn6/AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*,
_class"
 loc:@bn6/AssignMovingAvg/27037*
_output_shapes
:d2
bn6/AssignMovingAvg/mul?
'bn6/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpbn6_assignmovingavg_27037bn6/AssignMovingAvg/mul:z:0#^bn6/AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*,
_class"
 loc:@bn6/AssignMovingAvg/27037*
_output_shapes
 *
dtype02)
'bn6/AssignMovingAvg/AssignSubVariableOp?
bn6/AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*.
_class$
" loc:@bn6/AssignMovingAvg_1/27043*
_output_shapes
: *
dtype0*
valueB
 *???=2
bn6/AssignMovingAvg_1/decay?
$bn6/AssignMovingAvg_1/ReadVariableOpReadVariableOpbn6_assignmovingavg_1_27043*
_output_shapes
:d*
dtype02&
$bn6/AssignMovingAvg_1/ReadVariableOp?
bn6/AssignMovingAvg_1/subSub,bn6/AssignMovingAvg_1/ReadVariableOp:value:0bn6/moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*.
_class$
" loc:@bn6/AssignMovingAvg_1/27043*
_output_shapes
:d2
bn6/AssignMovingAvg_1/sub?
bn6/AssignMovingAvg_1/mulMulbn6/AssignMovingAvg_1/sub:z:0$bn6/AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*.
_class$
" loc:@bn6/AssignMovingAvg_1/27043*
_output_shapes
:d2
bn6/AssignMovingAvg_1/mul?
)bn6/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpbn6_assignmovingavg_1_27043bn6/AssignMovingAvg_1/mul:z:0%^bn6/AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*.
_class$
" loc:@bn6/AssignMovingAvg_1/27043*
_output_shapes
 *
dtype02+
)bn6/AssignMovingAvg_1/AssignSubVariableOpo
bn6/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *?7?52
bn6/batchnorm/add/y?
bn6/batchnorm/addAddV2bn6/moments/Squeeze_1:output:0bn6/batchnorm/add/y:output:0*
T0*
_output_shapes
:d2
bn6/batchnorm/addo
bn6/batchnorm/RsqrtRsqrtbn6/batchnorm/add:z:0*
T0*
_output_shapes
:d2
bn6/batchnorm/Rsqrt?
 bn6/batchnorm/mul/ReadVariableOpReadVariableOp)bn6_batchnorm_mul_readvariableop_resource*
_output_shapes
:d*
dtype02"
 bn6/batchnorm/mul/ReadVariableOp?
bn6/batchnorm/mulMulbn6/batchnorm/Rsqrt:y:0(bn6/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:d2
bn6/batchnorm/mul?
bn6/batchnorm/mul_1Muldense6/MatMul:product:0bn6/batchnorm/mul:z:0*
T0*'
_output_shapes
:?????????d2
bn6/batchnorm/mul_1?
bn6/batchnorm/mul_2Mulbn6/moments/Squeeze:output:0bn6/batchnorm/mul:z:0*
T0*
_output_shapes
:d2
bn6/batchnorm/mul_2?
bn6/batchnorm/ReadVariableOpReadVariableOp%bn6_batchnorm_readvariableop_resource*
_output_shapes
:d*
dtype02
bn6/batchnorm/ReadVariableOp?
bn6/batchnorm/subSub$bn6/batchnorm/ReadVariableOp:value:0bn6/batchnorm/mul_2:z:0*
T0*
_output_shapes
:d2
bn6/batchnorm/sub?
bn6/batchnorm/add_1AddV2bn6/batchnorm/mul_1:z:0bn6/batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????d2
bn6/batchnorm/add_1?
IdentityIdentitybn6/batchnorm/add_1:z:0#^batch_normalization/AssignNewValue%^batch_normalization/AssignNewValue_14^batch_normalization/FusedBatchNormV3/ReadVariableOp6^batch_normalization/FusedBatchNormV3/ReadVariableOp_1#^batch_normalization/ReadVariableOp%^batch_normalization/ReadVariableOp_1^binary_conv2d/ReadVariableOp^bn1/AssignNewValue^bn1/AssignNewValue_1$^bn1/FusedBatchNormV3/ReadVariableOp&^bn1/FusedBatchNormV3/ReadVariableOp_1^bn1/ReadVariableOp^bn1/ReadVariableOp_1(^bn5/AssignMovingAvg/AssignSubVariableOp#^bn5/AssignMovingAvg/ReadVariableOp*^bn5/AssignMovingAvg_1/AssignSubVariableOp%^bn5/AssignMovingAvg_1/ReadVariableOp^bn5/batchnorm/ReadVariableOp!^bn5/batchnorm/mul/ReadVariableOp(^bn6/AssignMovingAvg/AssignSubVariableOp#^bn6/AssignMovingAvg/ReadVariableOp*^bn6/AssignMovingAvg_1/AssignSubVariableOp%^bn6/AssignMovingAvg_1/ReadVariableOp^bn6/batchnorm/ReadVariableOp!^bn6/batchnorm/mul/ReadVariableOp^conv1/ReadVariableOp^dense5/ReadVariableOp^dense6/ReadVariableOp*
T0*'
_output_shapes
:?????????d2

Identity"
identityIdentity:output:0*~
_input_shapesm
k:?????????@@::::::::::::::::::::2H
"batch_normalization/AssignNewValue"batch_normalization/AssignNewValue2L
$batch_normalization/AssignNewValue_1$batch_normalization/AssignNewValue_12j
3batch_normalization/FusedBatchNormV3/ReadVariableOp3batch_normalization/FusedBatchNormV3/ReadVariableOp2n
5batch_normalization/FusedBatchNormV3/ReadVariableOp_15batch_normalization/FusedBatchNormV3/ReadVariableOp_12H
"batch_normalization/ReadVariableOp"batch_normalization/ReadVariableOp2L
$batch_normalization/ReadVariableOp_1$batch_normalization/ReadVariableOp_12<
binary_conv2d/ReadVariableOpbinary_conv2d/ReadVariableOp2(
bn1/AssignNewValuebn1/AssignNewValue2,
bn1/AssignNewValue_1bn1/AssignNewValue_12J
#bn1/FusedBatchNormV3/ReadVariableOp#bn1/FusedBatchNormV3/ReadVariableOp2N
%bn1/FusedBatchNormV3/ReadVariableOp_1%bn1/FusedBatchNormV3/ReadVariableOp_12(
bn1/ReadVariableOpbn1/ReadVariableOp2,
bn1/ReadVariableOp_1bn1/ReadVariableOp_12R
'bn5/AssignMovingAvg/AssignSubVariableOp'bn5/AssignMovingAvg/AssignSubVariableOp2H
"bn5/AssignMovingAvg/ReadVariableOp"bn5/AssignMovingAvg/ReadVariableOp2V
)bn5/AssignMovingAvg_1/AssignSubVariableOp)bn5/AssignMovingAvg_1/AssignSubVariableOp2L
$bn5/AssignMovingAvg_1/ReadVariableOp$bn5/AssignMovingAvg_1/ReadVariableOp2<
bn5/batchnorm/ReadVariableOpbn5/batchnorm/ReadVariableOp2D
 bn5/batchnorm/mul/ReadVariableOp bn5/batchnorm/mul/ReadVariableOp2R
'bn6/AssignMovingAvg/AssignSubVariableOp'bn6/AssignMovingAvg/AssignSubVariableOp2H
"bn6/AssignMovingAvg/ReadVariableOp"bn6/AssignMovingAvg/ReadVariableOp2V
)bn6/AssignMovingAvg_1/AssignSubVariableOp)bn6/AssignMovingAvg_1/AssignSubVariableOp2L
$bn6/AssignMovingAvg_1/ReadVariableOp$bn6/AssignMovingAvg_1/ReadVariableOp2<
bn6/batchnorm/ReadVariableOpbn6/batchnorm/ReadVariableOp2D
 bn6/batchnorm/mul/ReadVariableOp bn6/batchnorm/mul/ReadVariableOp2,
conv1/ReadVariableOpconv1/ReadVariableOp2.
dense5/ReadVariableOpdense5/ReadVariableOp2.
dense6/ReadVariableOpdense6/ReadVariableOp:W S
/
_output_shapes
:?????????@@
 
_user_specified_nameinputs
?
?
>__inference_bn5_layer_call_and_return_conditional_losses_25942

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identity??batchnorm/ReadVariableOp?batchnorm/ReadVariableOp_1?batchnorm/ReadVariableOp_2?batchnorm/mul/ReadVariableOp?
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:?*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *?7?52
batchnorm/add/y?
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:?2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:?2
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:?*
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:?2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:??????????2
batchnorm/mul_1?
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:?*
dtype02
batchnorm/ReadVariableOp_1?
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:?2
batchnorm/mul_2?
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:?*
dtype02
batchnorm/ReadVariableOp_2?
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:??????????2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:??????????::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
>__inference_bn1_layer_call_and_return_conditional_losses_27380

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+????????? ??????????????????: : : : :*
data_formatNCHW*
epsilon%??'7*
exponential_avg_factor%???=2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+????????? ??????????????????2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+????????? ??????????????????::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+????????? ??????????????????
 
_user_specified_nameinputs
?
?
>__inference_bn1_layer_call_and_return_conditional_losses_26120

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????   : : : : :*
data_formatNCHW*
epsilon%??'7*
exponential_avg_factor%???=2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:?????????   2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????   ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????   
 
_user_specified_nameinputs
?
d
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_25551

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
N__inference_batch_normalization_layer_call_and_return_conditional_losses_26243

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:????????? :::::*
data_formatNCHW*
epsilon%??'7*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:????????? ::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
k
%__inference_conv1_layer_call_fn_25545

inputs
unknown
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_conv1_layer_call_and_return_conditional_losses_255372
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*D
_input_shapes3
1:+???????????????????????????:22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
A__inference_dense5_layer_call_and_return_conditional_losses_27704

inputs
readvariableop_resource
identity??ReadVariableOpz
ReadVariableOpReadVariableOpreadvariableop_resource* 
_output_shapes
:
?@?*
dtype02
ReadVariableOp[
	truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
	truediv/yt
truedivRealDivReadVariableOp:value:0truediv/y:output:0*
T0* 
_output_shapes
:
?@?2	
truedivS
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
mul/xY
mulMulmul/x:output:0truediv:z:0*
T0* 
_output_shapes
:
?@?2
mulS
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
add/yW
addAddV2mul:z:0add/y:output:0*
T0* 
_output_shapes
:
?@?2
addw
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value/Minimum/y?
clip_by_value/MinimumMinimumadd:z:0 clip_by_value/Minimum/y:output:0*
T0* 
_output_shapes
:
?@?2
clip_by_value/Minimumg
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value/y?
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0* 
_output_shapes
:
?@?2
clip_by_valueU
RoundRoundclip_by_value:z:0*
T0* 
_output_shapes
:
?@?2
RoundZ
subSub	Round:y:0clip_by_value:z:0*
T0* 
_output_shapes
:
?@?2
sub`
StopGradientStopGradientsub:z:0*
T0* 
_output_shapes
:
?@?2
StopGradientl
add_1AddV2clip_by_value:z:0StopGradient:output:0*
T0* 
_output_shapes
:
?@?2
add_1W
mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @2	
mul_1/x]
mul_1Mulmul_1/x:output:0	add_1:z:0*
T0* 
_output_shapes
:
?@?2
mul_1W
sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2	
sub_1/y]
sub_1Sub	mul_1:z:0sub_1/y:output:0*
T0* 
_output_shapes
:
?@?2
sub_1W
mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2	
mul_2/x]
mul_2Mulmul_2/x:output:0	sub_1:z:0*
T0* 
_output_shapes
:
?@?2
mul_2`
MatMulMatMulinputs	mul_2:z:0*
T0*(
_output_shapes
:??????????2
MatMulv
IdentityIdentityMatMul:product:0^ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*+
_input_shapes
:??????????@:2 
ReadVariableOpReadVariableOp:P L
(
_output_shapes
:??????????@
 
_user_specified_nameinputs
?
s
-__inference_binary_conv2d_layer_call_fn_25697

inputs
unknown
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_binary_conv2d_layer_call_and_return_conditional_losses_256892
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*D
_input_shapes3
1:+??????????????????????????? :22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
@
$__inference_act5_layer_call_fn_27818

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *H
fCRA
?__inference_act5_layer_call_and_return_conditional_losses_264152
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
>__inference_bn1_layer_call_and_return_conditional_losses_25619

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+????????? ??????????????????: : : : :*
data_formatNCHW*
epsilon%??'7*
exponential_avg_factor%???=2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+????????? ??????????????????2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+????????? ??????????????????::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+????????? ??????????????????
 
_user_specified_nameinputs
?
^
B__inference_flatten_layer_call_and_return_conditional_losses_27672

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????    2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????@2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????@2

Identity"
identityIdentity:output:0*.
_input_shapes
:????????? :W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
??
?
 __inference__wrapped_model_25509
conv1_input,
(sequential_conv1_readvariableop_resource*
&sequential_bn1_readvariableop_resource,
(sequential_bn1_readvariableop_1_resource;
7sequential_bn1_fusedbatchnormv3_readvariableop_resource=
9sequential_bn1_fusedbatchnormv3_readvariableop_1_resource4
0sequential_binary_conv2d_readvariableop_resource:
6sequential_batch_normalization_readvariableop_resource<
8sequential_batch_normalization_readvariableop_1_resourceK
Gsequential_batch_normalization_fusedbatchnormv3_readvariableop_resourceM
Isequential_batch_normalization_fusedbatchnormv3_readvariableop_1_resource-
)sequential_dense5_readvariableop_resource4
0sequential_bn5_batchnorm_readvariableop_resource8
4sequential_bn5_batchnorm_mul_readvariableop_resource6
2sequential_bn5_batchnorm_readvariableop_1_resource6
2sequential_bn5_batchnorm_readvariableop_2_resource-
)sequential_dense6_readvariableop_resource4
0sequential_bn6_batchnorm_readvariableop_resource8
4sequential_bn6_batchnorm_mul_readvariableop_resource6
2sequential_bn6_batchnorm_readvariableop_1_resource6
2sequential_bn6_batchnorm_readvariableop_2_resource
identity??>sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp?@sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1?-sequential/batch_normalization/ReadVariableOp?/sequential/batch_normalization/ReadVariableOp_1?'sequential/binary_conv2d/ReadVariableOp?.sequential/bn1/FusedBatchNormV3/ReadVariableOp?0sequential/bn1/FusedBatchNormV3/ReadVariableOp_1?sequential/bn1/ReadVariableOp?sequential/bn1/ReadVariableOp_1?'sequential/bn5/batchnorm/ReadVariableOp?)sequential/bn5/batchnorm/ReadVariableOp_1?)sequential/bn5/batchnorm/ReadVariableOp_2?+sequential/bn5/batchnorm/mul/ReadVariableOp?'sequential/bn6/batchnorm/ReadVariableOp?)sequential/bn6/batchnorm/ReadVariableOp_1?)sequential/bn6/batchnorm/ReadVariableOp_2?+sequential/bn6/batchnorm/mul/ReadVariableOp?sequential/conv1/ReadVariableOp? sequential/dense5/ReadVariableOp? sequential/dense6/ReadVariableOp?
sequential/conv1/ReadVariableOpReadVariableOp(sequential_conv1_readvariableop_resource*&
_output_shapes
: *
dtype02!
sequential/conv1/ReadVariableOp}
sequential/conv1/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
sequential/conv1/truediv/y?
sequential/conv1/truedivRealDiv'sequential/conv1/ReadVariableOp:value:0#sequential/conv1/truediv/y:output:0*
T0*&
_output_shapes
: 2
sequential/conv1/truedivu
sequential/conv1/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
sequential/conv1/mul/x?
sequential/conv1/mulMulsequential/conv1/mul/x:output:0sequential/conv1/truediv:z:0*
T0*&
_output_shapes
: 2
sequential/conv1/mulu
sequential/conv1/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
sequential/conv1/add/y?
sequential/conv1/addAddV2sequential/conv1/mul:z:0sequential/conv1/add/y:output:0*
T0*&
_output_shapes
: 2
sequential/conv1/add?
(sequential/conv1/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2*
(sequential/conv1/clip_by_value/Minimum/y?
&sequential/conv1/clip_by_value/MinimumMinimumsequential/conv1/add:z:01sequential/conv1/clip_by_value/Minimum/y:output:0*
T0*&
_output_shapes
: 2(
&sequential/conv1/clip_by_value/Minimum?
 sequential/conv1/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 sequential/conv1/clip_by_value/y?
sequential/conv1/clip_by_valueMaximum*sequential/conv1/clip_by_value/Minimum:z:0)sequential/conv1/clip_by_value/y:output:0*
T0*&
_output_shapes
: 2 
sequential/conv1/clip_by_value?
sequential/conv1/RoundRound"sequential/conv1/clip_by_value:z:0*
T0*&
_output_shapes
: 2
sequential/conv1/Round?
sequential/conv1/subSubsequential/conv1/Round:y:0"sequential/conv1/clip_by_value:z:0*
T0*&
_output_shapes
: 2
sequential/conv1/sub?
sequential/conv1/StopGradientStopGradientsequential/conv1/sub:z:0*
T0*&
_output_shapes
: 2
sequential/conv1/StopGradient?
sequential/conv1/add_1AddV2"sequential/conv1/clip_by_value:z:0&sequential/conv1/StopGradient:output:0*
T0*&
_output_shapes
: 2
sequential/conv1/add_1y
sequential/conv1/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
sequential/conv1/mul_1/x?
sequential/conv1/mul_1Mul!sequential/conv1/mul_1/x:output:0sequential/conv1/add_1:z:0*
T0*&
_output_shapes
: 2
sequential/conv1/mul_1y
sequential/conv1/sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
sequential/conv1/sub_1/y?
sequential/conv1/sub_1Subsequential/conv1/mul_1:z:0!sequential/conv1/sub_1/y:output:0*
T0*&
_output_shapes
: 2
sequential/conv1/sub_1y
sequential/conv1/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
sequential/conv1/mul_2/x?
sequential/conv1/mul_2Mul!sequential/conv1/mul_2/x:output:0sequential/conv1/sub_1:z:0*
T0*&
_output_shapes
: 2
sequential/conv1/mul_2?
sequential/conv1/convolutionConv2Dconv1_inputsequential/conv1/mul_2:z:0*
T0*/
_output_shapes
:?????????@@ *
paddingSAME*
strides
2
sequential/conv1/convolution?
 sequential/max_pooling2d/MaxPoolMaxPool%sequential/conv1/convolution:output:0*/
_output_shapes
:?????????   *
ksize
*
paddingVALID*
strides
2"
 sequential/max_pooling2d/MaxPool?
sequential/bn1/ReadVariableOpReadVariableOp&sequential_bn1_readvariableop_resource*
_output_shapes
: *
dtype02
sequential/bn1/ReadVariableOp?
sequential/bn1/ReadVariableOp_1ReadVariableOp(sequential_bn1_readvariableop_1_resource*
_output_shapes
: *
dtype02!
sequential/bn1/ReadVariableOp_1?
.sequential/bn1/FusedBatchNormV3/ReadVariableOpReadVariableOp7sequential_bn1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype020
.sequential/bn1/FusedBatchNormV3/ReadVariableOp?
0sequential/bn1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp9sequential_bn1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype022
0sequential/bn1/FusedBatchNormV3/ReadVariableOp_1?
sequential/bn1/FusedBatchNormV3FusedBatchNormV3)sequential/max_pooling2d/MaxPool:output:0%sequential/bn1/ReadVariableOp:value:0'sequential/bn1/ReadVariableOp_1:value:06sequential/bn1/FusedBatchNormV3/ReadVariableOp:value:08sequential/bn1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????   : : : : :*
data_formatNCHW*
epsilon%??'7*
is_training( 2!
sequential/bn1/FusedBatchNormV3s
sequential/act1/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
sequential/act1/mul/x?
sequential/act1/mulMulsequential/act1/mul/x:output:0#sequential/bn1/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????   2
sequential/act1/muls
sequential/act1/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
sequential/act1/add/y?
sequential/act1/addAddV2sequential/act1/mul:z:0sequential/act1/add/y:output:0*
T0*/
_output_shapes
:?????????   2
sequential/act1/add?
'sequential/act1/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2)
'sequential/act1/clip_by_value/Minimum/y?
%sequential/act1/clip_by_value/MinimumMinimumsequential/act1/add:z:00sequential/act1/clip_by_value/Minimum/y:output:0*
T0*/
_output_shapes
:?????????   2'
%sequential/act1/clip_by_value/Minimum?
sequential/act1/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
sequential/act1/clip_by_value/y?
sequential/act1/clip_by_valueMaximum)sequential/act1/clip_by_value/Minimum:z:0(sequential/act1/clip_by_value/y:output:0*
T0*/
_output_shapes
:?????????   2
sequential/act1/clip_by_value?
sequential/act1/RoundRound!sequential/act1/clip_by_value:z:0*
T0*/
_output_shapes
:?????????   2
sequential/act1/Round?
sequential/act1/subSubsequential/act1/Round:y:0!sequential/act1/clip_by_value:z:0*
T0*/
_output_shapes
:?????????   2
sequential/act1/sub?
sequential/act1/StopGradientStopGradientsequential/act1/sub:z:0*
T0*/
_output_shapes
:?????????   2
sequential/act1/StopGradient?
sequential/act1/add_1AddV2!sequential/act1/clip_by_value:z:0%sequential/act1/StopGradient:output:0*
T0*/
_output_shapes
:?????????   2
sequential/act1/add_1w
sequential/act1/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
sequential/act1/mul_1/x?
sequential/act1/mul_1Mul sequential/act1/mul_1/x:output:0sequential/act1/add_1:z:0*
T0*/
_output_shapes
:?????????   2
sequential/act1/mul_1w
sequential/act1/sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
sequential/act1/sub_1/y?
sequential/act1/sub_1Subsequential/act1/mul_1:z:0 sequential/act1/sub_1/y:output:0*
T0*/
_output_shapes
:?????????   2
sequential/act1/sub_1?
'sequential/binary_conv2d/ReadVariableOpReadVariableOp0sequential_binary_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02)
'sequential/binary_conv2d/ReadVariableOp?
"sequential/binary_conv2d/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2$
"sequential/binary_conv2d/truediv/y?
 sequential/binary_conv2d/truedivRealDiv/sequential/binary_conv2d/ReadVariableOp:value:0+sequential/binary_conv2d/truediv/y:output:0*
T0*&
_output_shapes
:  2"
 sequential/binary_conv2d/truediv?
sequential/binary_conv2d/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2 
sequential/binary_conv2d/mul/x?
sequential/binary_conv2d/mulMul'sequential/binary_conv2d/mul/x:output:0$sequential/binary_conv2d/truediv:z:0*
T0*&
_output_shapes
:  2
sequential/binary_conv2d/mul?
sequential/binary_conv2d/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2 
sequential/binary_conv2d/add/y?
sequential/binary_conv2d/addAddV2 sequential/binary_conv2d/mul:z:0'sequential/binary_conv2d/add/y:output:0*
T0*&
_output_shapes
:  2
sequential/binary_conv2d/add?
0sequential/binary_conv2d/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??22
0sequential/binary_conv2d/clip_by_value/Minimum/y?
.sequential/binary_conv2d/clip_by_value/MinimumMinimum sequential/binary_conv2d/add:z:09sequential/binary_conv2d/clip_by_value/Minimum/y:output:0*
T0*&
_output_shapes
:  20
.sequential/binary_conv2d/clip_by_value/Minimum?
(sequential/binary_conv2d/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(sequential/binary_conv2d/clip_by_value/y?
&sequential/binary_conv2d/clip_by_valueMaximum2sequential/binary_conv2d/clip_by_value/Minimum:z:01sequential/binary_conv2d/clip_by_value/y:output:0*
T0*&
_output_shapes
:  2(
&sequential/binary_conv2d/clip_by_value?
sequential/binary_conv2d/RoundRound*sequential/binary_conv2d/clip_by_value:z:0*
T0*&
_output_shapes
:  2 
sequential/binary_conv2d/Round?
sequential/binary_conv2d/subSub"sequential/binary_conv2d/Round:y:0*sequential/binary_conv2d/clip_by_value:z:0*
T0*&
_output_shapes
:  2
sequential/binary_conv2d/sub?
%sequential/binary_conv2d/StopGradientStopGradient sequential/binary_conv2d/sub:z:0*
T0*&
_output_shapes
:  2'
%sequential/binary_conv2d/StopGradient?
sequential/binary_conv2d/add_1AddV2*sequential/binary_conv2d/clip_by_value:z:0.sequential/binary_conv2d/StopGradient:output:0*
T0*&
_output_shapes
:  2 
sequential/binary_conv2d/add_1?
 sequential/binary_conv2d/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @2"
 sequential/binary_conv2d/mul_1/x?
sequential/binary_conv2d/mul_1Mul)sequential/binary_conv2d/mul_1/x:output:0"sequential/binary_conv2d/add_1:z:0*
T0*&
_output_shapes
:  2 
sequential/binary_conv2d/mul_1?
 sequential/binary_conv2d/sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2"
 sequential/binary_conv2d/sub_1/y?
sequential/binary_conv2d/sub_1Sub"sequential/binary_conv2d/mul_1:z:0)sequential/binary_conv2d/sub_1/y:output:0*
T0*&
_output_shapes
:  2 
sequential/binary_conv2d/sub_1?
 sequential/binary_conv2d/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2"
 sequential/binary_conv2d/mul_2/x?
sequential/binary_conv2d/mul_2Mul)sequential/binary_conv2d/mul_2/x:output:0"sequential/binary_conv2d/sub_1:z:0*
T0*&
_output_shapes
:  2 
sequential/binary_conv2d/mul_2?
$sequential/binary_conv2d/convolutionConv2Dsequential/act1/sub_1:z:0"sequential/binary_conv2d/mul_2:z:0*
T0*/
_output_shapes
:?????????   *
paddingSAME*
strides
2&
$sequential/binary_conv2d/convolution?
"sequential/max_pooling2d_1/MaxPoolMaxPool-sequential/binary_conv2d/convolution:output:0*/
_output_shapes
:????????? *
ksize
*
paddingVALID*
strides
2$
"sequential/max_pooling2d_1/MaxPool?
-sequential/batch_normalization/ReadVariableOpReadVariableOp6sequential_batch_normalization_readvariableop_resource*
_output_shapes
:*
dtype02/
-sequential/batch_normalization/ReadVariableOp?
/sequential/batch_normalization/ReadVariableOp_1ReadVariableOp8sequential_batch_normalization_readvariableop_1_resource*
_output_shapes
:*
dtype021
/sequential/batch_normalization/ReadVariableOp_1?
>sequential/batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOpGsequential_batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02@
>sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp?
@sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpIsequential_batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02B
@sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1?
/sequential/batch_normalization/FusedBatchNormV3FusedBatchNormV3+sequential/max_pooling2d_1/MaxPool:output:05sequential/batch_normalization/ReadVariableOp:value:07sequential/batch_normalization/ReadVariableOp_1:value:0Fsequential/batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0Hsequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:????????? :::::*
data_formatNCHW*
epsilon%??'7*
is_training( 21
/sequential/batch_normalization/FusedBatchNormV3
sequential/activation/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
sequential/activation/mul/x?
sequential/activation/mulMul$sequential/activation/mul/x:output:03sequential/batch_normalization/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:????????? 2
sequential/activation/mul
sequential/activation/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
sequential/activation/add/y?
sequential/activation/addAddV2sequential/activation/mul:z:0$sequential/activation/add/y:output:0*
T0*/
_output_shapes
:????????? 2
sequential/activation/add?
-sequential/activation/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2/
-sequential/activation/clip_by_value/Minimum/y?
+sequential/activation/clip_by_value/MinimumMinimumsequential/activation/add:z:06sequential/activation/clip_by_value/Minimum/y:output:0*
T0*/
_output_shapes
:????????? 2-
+sequential/activation/clip_by_value/Minimum?
%sequential/activation/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2'
%sequential/activation/clip_by_value/y?
#sequential/activation/clip_by_valueMaximum/sequential/activation/clip_by_value/Minimum:z:0.sequential/activation/clip_by_value/y:output:0*
T0*/
_output_shapes
:????????? 2%
#sequential/activation/clip_by_value?
sequential/activation/RoundRound'sequential/activation/clip_by_value:z:0*
T0*/
_output_shapes
:????????? 2
sequential/activation/Round?
sequential/activation/subSubsequential/activation/Round:y:0'sequential/activation/clip_by_value:z:0*
T0*/
_output_shapes
:????????? 2
sequential/activation/sub?
"sequential/activation/StopGradientStopGradientsequential/activation/sub:z:0*
T0*/
_output_shapes
:????????? 2$
"sequential/activation/StopGradient?
sequential/activation/add_1AddV2'sequential/activation/clip_by_value:z:0+sequential/activation/StopGradient:output:0*
T0*/
_output_shapes
:????????? 2
sequential/activation/add_1?
sequential/activation/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
sequential/activation/mul_1/x?
sequential/activation/mul_1Mul&sequential/activation/mul_1/x:output:0sequential/activation/add_1:z:0*
T0*/
_output_shapes
:????????? 2
sequential/activation/mul_1?
sequential/activation/sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
sequential/activation/sub_1/y?
sequential/activation/sub_1Subsequential/activation/mul_1:z:0&sequential/activation/sub_1/y:output:0*
T0*/
_output_shapes
:????????? 2
sequential/activation/sub_1?
sequential/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????    2
sequential/flatten/Const?
sequential/flatten/ReshapeReshapesequential/activation/sub_1:z:0!sequential/flatten/Const:output:0*
T0*(
_output_shapes
:??????????@2
sequential/flatten/Reshape?
 sequential/dense5/ReadVariableOpReadVariableOp)sequential_dense5_readvariableop_resource* 
_output_shapes
:
?@?*
dtype02"
 sequential/dense5/ReadVariableOp
sequential/dense5/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
sequential/dense5/truediv/y?
sequential/dense5/truedivRealDiv(sequential/dense5/ReadVariableOp:value:0$sequential/dense5/truediv/y:output:0*
T0* 
_output_shapes
:
?@?2
sequential/dense5/truedivw
sequential/dense5/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
sequential/dense5/mul/x?
sequential/dense5/mulMul sequential/dense5/mul/x:output:0sequential/dense5/truediv:z:0*
T0* 
_output_shapes
:
?@?2
sequential/dense5/mulw
sequential/dense5/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
sequential/dense5/add/y?
sequential/dense5/addAddV2sequential/dense5/mul:z:0 sequential/dense5/add/y:output:0*
T0* 
_output_shapes
:
?@?2
sequential/dense5/add?
)sequential/dense5/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2+
)sequential/dense5/clip_by_value/Minimum/y?
'sequential/dense5/clip_by_value/MinimumMinimumsequential/dense5/add:z:02sequential/dense5/clip_by_value/Minimum/y:output:0*
T0* 
_output_shapes
:
?@?2)
'sequential/dense5/clip_by_value/Minimum?
!sequential/dense5/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!sequential/dense5/clip_by_value/y?
sequential/dense5/clip_by_valueMaximum+sequential/dense5/clip_by_value/Minimum:z:0*sequential/dense5/clip_by_value/y:output:0*
T0* 
_output_shapes
:
?@?2!
sequential/dense5/clip_by_value?
sequential/dense5/RoundRound#sequential/dense5/clip_by_value:z:0*
T0* 
_output_shapes
:
?@?2
sequential/dense5/Round?
sequential/dense5/subSubsequential/dense5/Round:y:0#sequential/dense5/clip_by_value:z:0*
T0* 
_output_shapes
:
?@?2
sequential/dense5/sub?
sequential/dense5/StopGradientStopGradientsequential/dense5/sub:z:0*
T0* 
_output_shapes
:
?@?2 
sequential/dense5/StopGradient?
sequential/dense5/add_1AddV2#sequential/dense5/clip_by_value:z:0'sequential/dense5/StopGradient:output:0*
T0* 
_output_shapes
:
?@?2
sequential/dense5/add_1{
sequential/dense5/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
sequential/dense5/mul_1/x?
sequential/dense5/mul_1Mul"sequential/dense5/mul_1/x:output:0sequential/dense5/add_1:z:0*
T0* 
_output_shapes
:
?@?2
sequential/dense5/mul_1{
sequential/dense5/sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
sequential/dense5/sub_1/y?
sequential/dense5/sub_1Subsequential/dense5/mul_1:z:0"sequential/dense5/sub_1/y:output:0*
T0* 
_output_shapes
:
?@?2
sequential/dense5/sub_1{
sequential/dense5/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
sequential/dense5/mul_2/x?
sequential/dense5/mul_2Mul"sequential/dense5/mul_2/x:output:0sequential/dense5/sub_1:z:0*
T0* 
_output_shapes
:
?@?2
sequential/dense5/mul_2?
sequential/dense5/MatMulMatMul#sequential/flatten/Reshape:output:0sequential/dense5/mul_2:z:0*
T0*(
_output_shapes
:??????????2
sequential/dense5/MatMul?
'sequential/bn5/batchnorm/ReadVariableOpReadVariableOp0sequential_bn5_batchnorm_readvariableop_resource*
_output_shapes	
:?*
dtype02)
'sequential/bn5/batchnorm/ReadVariableOp?
sequential/bn5/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *?7?52 
sequential/bn5/batchnorm/add/y?
sequential/bn5/batchnorm/addAddV2/sequential/bn5/batchnorm/ReadVariableOp:value:0'sequential/bn5/batchnorm/add/y:output:0*
T0*
_output_shapes	
:?2
sequential/bn5/batchnorm/add?
sequential/bn5/batchnorm/RsqrtRsqrt sequential/bn5/batchnorm/add:z:0*
T0*
_output_shapes	
:?2 
sequential/bn5/batchnorm/Rsqrt?
+sequential/bn5/batchnorm/mul/ReadVariableOpReadVariableOp4sequential_bn5_batchnorm_mul_readvariableop_resource*
_output_shapes	
:?*
dtype02-
+sequential/bn5/batchnorm/mul/ReadVariableOp?
sequential/bn5/batchnorm/mulMul"sequential/bn5/batchnorm/Rsqrt:y:03sequential/bn5/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:?2
sequential/bn5/batchnorm/mul?
sequential/bn5/batchnorm/mul_1Mul"sequential/dense5/MatMul:product:0 sequential/bn5/batchnorm/mul:z:0*
T0*(
_output_shapes
:??????????2 
sequential/bn5/batchnorm/mul_1?
)sequential/bn5/batchnorm/ReadVariableOp_1ReadVariableOp2sequential_bn5_batchnorm_readvariableop_1_resource*
_output_shapes	
:?*
dtype02+
)sequential/bn5/batchnorm/ReadVariableOp_1?
sequential/bn5/batchnorm/mul_2Mul1sequential/bn5/batchnorm/ReadVariableOp_1:value:0 sequential/bn5/batchnorm/mul:z:0*
T0*
_output_shapes	
:?2 
sequential/bn5/batchnorm/mul_2?
)sequential/bn5/batchnorm/ReadVariableOp_2ReadVariableOp2sequential_bn5_batchnorm_readvariableop_2_resource*
_output_shapes	
:?*
dtype02+
)sequential/bn5/batchnorm/ReadVariableOp_2?
sequential/bn5/batchnorm/subSub1sequential/bn5/batchnorm/ReadVariableOp_2:value:0"sequential/bn5/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?2
sequential/bn5/batchnorm/sub?
sequential/bn5/batchnorm/add_1AddV2"sequential/bn5/batchnorm/mul_1:z:0 sequential/bn5/batchnorm/sub:z:0*
T0*(
_output_shapes
:??????????2 
sequential/bn5/batchnorm/add_1s
sequential/act5/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
sequential/act5/mul/x?
sequential/act5/mulMulsequential/act5/mul/x:output:0"sequential/bn5/batchnorm/add_1:z:0*
T0*(
_output_shapes
:??????????2
sequential/act5/muls
sequential/act5/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
sequential/act5/add/y?
sequential/act5/addAddV2sequential/act5/mul:z:0sequential/act5/add/y:output:0*
T0*(
_output_shapes
:??????????2
sequential/act5/add?
'sequential/act5/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2)
'sequential/act5/clip_by_value/Minimum/y?
%sequential/act5/clip_by_value/MinimumMinimumsequential/act5/add:z:00sequential/act5/clip_by_value/Minimum/y:output:0*
T0*(
_output_shapes
:??????????2'
%sequential/act5/clip_by_value/Minimum?
sequential/act5/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
sequential/act5/clip_by_value/y?
sequential/act5/clip_by_valueMaximum)sequential/act5/clip_by_value/Minimum:z:0(sequential/act5/clip_by_value/y:output:0*
T0*(
_output_shapes
:??????????2
sequential/act5/clip_by_value?
sequential/act5/RoundRound!sequential/act5/clip_by_value:z:0*
T0*(
_output_shapes
:??????????2
sequential/act5/Round?
sequential/act5/subSubsequential/act5/Round:y:0!sequential/act5/clip_by_value:z:0*
T0*(
_output_shapes
:??????????2
sequential/act5/sub?
sequential/act5/StopGradientStopGradientsequential/act5/sub:z:0*
T0*(
_output_shapes
:??????????2
sequential/act5/StopGradient?
sequential/act5/add_1AddV2!sequential/act5/clip_by_value:z:0%sequential/act5/StopGradient:output:0*
T0*(
_output_shapes
:??????????2
sequential/act5/add_1w
sequential/act5/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
sequential/act5/mul_1/x?
sequential/act5/mul_1Mul sequential/act5/mul_1/x:output:0sequential/act5/add_1:z:0*
T0*(
_output_shapes
:??????????2
sequential/act5/mul_1w
sequential/act5/sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
sequential/act5/sub_1/y?
sequential/act5/sub_1Subsequential/act5/mul_1:z:0 sequential/act5/sub_1/y:output:0*
T0*(
_output_shapes
:??????????2
sequential/act5/sub_1?
 sequential/dense6/ReadVariableOpReadVariableOp)sequential_dense6_readvariableop_resource*
_output_shapes
:	?d*
dtype02"
 sequential/dense6/ReadVariableOp
sequential/dense6/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
sequential/dense6/truediv/y?
sequential/dense6/truedivRealDiv(sequential/dense6/ReadVariableOp:value:0$sequential/dense6/truediv/y:output:0*
T0*
_output_shapes
:	?d2
sequential/dense6/truedivw
sequential/dense6/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
sequential/dense6/mul/x?
sequential/dense6/mulMul sequential/dense6/mul/x:output:0sequential/dense6/truediv:z:0*
T0*
_output_shapes
:	?d2
sequential/dense6/mulw
sequential/dense6/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
sequential/dense6/add/y?
sequential/dense6/addAddV2sequential/dense6/mul:z:0 sequential/dense6/add/y:output:0*
T0*
_output_shapes
:	?d2
sequential/dense6/add?
)sequential/dense6/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2+
)sequential/dense6/clip_by_value/Minimum/y?
'sequential/dense6/clip_by_value/MinimumMinimumsequential/dense6/add:z:02sequential/dense6/clip_by_value/Minimum/y:output:0*
T0*
_output_shapes
:	?d2)
'sequential/dense6/clip_by_value/Minimum?
!sequential/dense6/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!sequential/dense6/clip_by_value/y?
sequential/dense6/clip_by_valueMaximum+sequential/dense6/clip_by_value/Minimum:z:0*sequential/dense6/clip_by_value/y:output:0*
T0*
_output_shapes
:	?d2!
sequential/dense6/clip_by_value?
sequential/dense6/RoundRound#sequential/dense6/clip_by_value:z:0*
T0*
_output_shapes
:	?d2
sequential/dense6/Round?
sequential/dense6/subSubsequential/dense6/Round:y:0#sequential/dense6/clip_by_value:z:0*
T0*
_output_shapes
:	?d2
sequential/dense6/sub?
sequential/dense6/StopGradientStopGradientsequential/dense6/sub:z:0*
T0*
_output_shapes
:	?d2 
sequential/dense6/StopGradient?
sequential/dense6/add_1AddV2#sequential/dense6/clip_by_value:z:0'sequential/dense6/StopGradient:output:0*
T0*
_output_shapes
:	?d2
sequential/dense6/add_1{
sequential/dense6/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
sequential/dense6/mul_1/x?
sequential/dense6/mul_1Mul"sequential/dense6/mul_1/x:output:0sequential/dense6/add_1:z:0*
T0*
_output_shapes
:	?d2
sequential/dense6/mul_1{
sequential/dense6/sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
sequential/dense6/sub_1/y?
sequential/dense6/sub_1Subsequential/dense6/mul_1:z:0"sequential/dense6/sub_1/y:output:0*
T0*
_output_shapes
:	?d2
sequential/dense6/sub_1{
sequential/dense6/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
sequential/dense6/mul_2/x?
sequential/dense6/mul_2Mul"sequential/dense6/mul_2/x:output:0sequential/dense6/sub_1:z:0*
T0*
_output_shapes
:	?d2
sequential/dense6/mul_2?
sequential/dense6/MatMulMatMulsequential/act5/sub_1:z:0sequential/dense6/mul_2:z:0*
T0*'
_output_shapes
:?????????d2
sequential/dense6/MatMul?
'sequential/bn6/batchnorm/ReadVariableOpReadVariableOp0sequential_bn6_batchnorm_readvariableop_resource*
_output_shapes
:d*
dtype02)
'sequential/bn6/batchnorm/ReadVariableOp?
sequential/bn6/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *?7?52 
sequential/bn6/batchnorm/add/y?
sequential/bn6/batchnorm/addAddV2/sequential/bn6/batchnorm/ReadVariableOp:value:0'sequential/bn6/batchnorm/add/y:output:0*
T0*
_output_shapes
:d2
sequential/bn6/batchnorm/add?
sequential/bn6/batchnorm/RsqrtRsqrt sequential/bn6/batchnorm/add:z:0*
T0*
_output_shapes
:d2 
sequential/bn6/batchnorm/Rsqrt?
+sequential/bn6/batchnorm/mul/ReadVariableOpReadVariableOp4sequential_bn6_batchnorm_mul_readvariableop_resource*
_output_shapes
:d*
dtype02-
+sequential/bn6/batchnorm/mul/ReadVariableOp?
sequential/bn6/batchnorm/mulMul"sequential/bn6/batchnorm/Rsqrt:y:03sequential/bn6/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:d2
sequential/bn6/batchnorm/mul?
sequential/bn6/batchnorm/mul_1Mul"sequential/dense6/MatMul:product:0 sequential/bn6/batchnorm/mul:z:0*
T0*'
_output_shapes
:?????????d2 
sequential/bn6/batchnorm/mul_1?
)sequential/bn6/batchnorm/ReadVariableOp_1ReadVariableOp2sequential_bn6_batchnorm_readvariableop_1_resource*
_output_shapes
:d*
dtype02+
)sequential/bn6/batchnorm/ReadVariableOp_1?
sequential/bn6/batchnorm/mul_2Mul1sequential/bn6/batchnorm/ReadVariableOp_1:value:0 sequential/bn6/batchnorm/mul:z:0*
T0*
_output_shapes
:d2 
sequential/bn6/batchnorm/mul_2?
)sequential/bn6/batchnorm/ReadVariableOp_2ReadVariableOp2sequential_bn6_batchnorm_readvariableop_2_resource*
_output_shapes
:d*
dtype02+
)sequential/bn6/batchnorm/ReadVariableOp_2?
sequential/bn6/batchnorm/subSub1sequential/bn6/batchnorm/ReadVariableOp_2:value:0"sequential/bn6/batchnorm/mul_2:z:0*
T0*
_output_shapes
:d2
sequential/bn6/batchnorm/sub?
sequential/bn6/batchnorm/add_1AddV2"sequential/bn6/batchnorm/mul_1:z:0 sequential/bn6/batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????d2 
sequential/bn6/batchnorm/add_1?
IdentityIdentity"sequential/bn6/batchnorm/add_1:z:0?^sequential/batch_normalization/FusedBatchNormV3/ReadVariableOpA^sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1.^sequential/batch_normalization/ReadVariableOp0^sequential/batch_normalization/ReadVariableOp_1(^sequential/binary_conv2d/ReadVariableOp/^sequential/bn1/FusedBatchNormV3/ReadVariableOp1^sequential/bn1/FusedBatchNormV3/ReadVariableOp_1^sequential/bn1/ReadVariableOp ^sequential/bn1/ReadVariableOp_1(^sequential/bn5/batchnorm/ReadVariableOp*^sequential/bn5/batchnorm/ReadVariableOp_1*^sequential/bn5/batchnorm/ReadVariableOp_2,^sequential/bn5/batchnorm/mul/ReadVariableOp(^sequential/bn6/batchnorm/ReadVariableOp*^sequential/bn6/batchnorm/ReadVariableOp_1*^sequential/bn6/batchnorm/ReadVariableOp_2,^sequential/bn6/batchnorm/mul/ReadVariableOp ^sequential/conv1/ReadVariableOp!^sequential/dense5/ReadVariableOp!^sequential/dense6/ReadVariableOp*
T0*'
_output_shapes
:?????????d2

Identity"
identityIdentity:output:0*~
_input_shapesm
k:?????????@@::::::::::::::::::::2?
>sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp>sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp2?
@sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1@sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_12^
-sequential/batch_normalization/ReadVariableOp-sequential/batch_normalization/ReadVariableOp2b
/sequential/batch_normalization/ReadVariableOp_1/sequential/batch_normalization/ReadVariableOp_12R
'sequential/binary_conv2d/ReadVariableOp'sequential/binary_conv2d/ReadVariableOp2`
.sequential/bn1/FusedBatchNormV3/ReadVariableOp.sequential/bn1/FusedBatchNormV3/ReadVariableOp2d
0sequential/bn1/FusedBatchNormV3/ReadVariableOp_10sequential/bn1/FusedBatchNormV3/ReadVariableOp_12>
sequential/bn1/ReadVariableOpsequential/bn1/ReadVariableOp2B
sequential/bn1/ReadVariableOp_1sequential/bn1/ReadVariableOp_12R
'sequential/bn5/batchnorm/ReadVariableOp'sequential/bn5/batchnorm/ReadVariableOp2V
)sequential/bn5/batchnorm/ReadVariableOp_1)sequential/bn5/batchnorm/ReadVariableOp_12V
)sequential/bn5/batchnorm/ReadVariableOp_2)sequential/bn5/batchnorm/ReadVariableOp_22Z
+sequential/bn5/batchnorm/mul/ReadVariableOp+sequential/bn5/batchnorm/mul/ReadVariableOp2R
'sequential/bn6/batchnorm/ReadVariableOp'sequential/bn6/batchnorm/ReadVariableOp2V
)sequential/bn6/batchnorm/ReadVariableOp_1)sequential/bn6/batchnorm/ReadVariableOp_12V
)sequential/bn6/batchnorm/ReadVariableOp_2)sequential/bn6/batchnorm/ReadVariableOp_22Z
+sequential/bn6/batchnorm/mul/ReadVariableOp+sequential/bn6/batchnorm/mul/ReadVariableOp2B
sequential/conv1/ReadVariableOpsequential/conv1/ReadVariableOp2D
 sequential/dense5/ReadVariableOp sequential/dense5/ReadVariableOp2D
 sequential/dense6/ReadVariableOp sequential/dense6/ReadVariableOp:\ X
/
_output_shapes
:?????????@@
%
_user_specified_nameconv1_input
?
l
&__inference_dense6_layer_call_fn_27852

inputs
unknown
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_dense6_layer_call_and_return_conditional_losses_264502
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????d2

Identity"
identityIdentity:output:0*+
_input_shapes
:??????????:22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?/
?
>__inference_bn5_layer_call_and_return_conditional_losses_27747

inputs
assignmovingavg_27722
assignmovingavg_1_27728)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identity??#AssignMovingAvg/AssignSubVariableOp?AssignMovingAvg/ReadVariableOp?%AssignMovingAvg_1/AssignSubVariableOp? AssignMovingAvg_1/ReadVariableOp?batchnorm/ReadVariableOp?batchnorm/mul/ReadVariableOp?
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indices?
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	?*
	keep_dims(2
moments/mean}
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	?2
moments/StopGradient?
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:??????????2
moments/SquaredDifference?
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	?*
	keep_dims(2
moments/variance?
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:?*
squeeze_dims
 2
moments/Squeeze?
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:?*
squeeze_dims
 2
moments/Squeeze_1?
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*(
_class
loc:@AssignMovingAvg/27722*
_output_shapes
: *
dtype0*
valueB
 *???=2
AssignMovingAvg/decay?
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_27722*
_output_shapes	
:?*
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*(
_class
loc:@AssignMovingAvg/27722*
_output_shapes	
:?2
AssignMovingAvg/sub?
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*(
_class
loc:@AssignMovingAvg/27722*
_output_shapes	
:?2
AssignMovingAvg/mul?
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_27722AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*(
_class
loc:@AssignMovingAvg/27722*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp?
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0**
_class 
loc:@AssignMovingAvg_1/27728*
_output_shapes
: *
dtype0*
valueB
 *???=2
AssignMovingAvg_1/decay?
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_27728*
_output_shapes	
:?*
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0**
_class 
loc:@AssignMovingAvg_1/27728*
_output_shapes	
:?2
AssignMovingAvg_1/sub?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0**
_class 
loc:@AssignMovingAvg_1/27728*
_output_shapes	
:?2
AssignMovingAvg_1/mul?
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_27728AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0**
_class 
loc:@AssignMovingAvg_1/27728*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *?7?52
batchnorm/add/y?
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:?2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:?2
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:?*
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:?2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:??????????2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:?2
batchnorm/mul_2?
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:?*
dtype02
batchnorm/ReadVariableOp?
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:??????????2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:??????????::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
>__inference_bn6_layer_call_and_return_conditional_losses_27908

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identity??batchnorm/ReadVariableOp?batchnorm/ReadVariableOp_1?batchnorm/ReadVariableOp_2?batchnorm/mul/ReadVariableOp?
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:d*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *?7?52
batchnorm/add/y?
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:d2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:d2
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:d*
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:d2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:?????????d2
batchnorm/mul_1?
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:d*
dtype02
batchnorm/ReadVariableOp_1?
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:d2
batchnorm/mul_2?
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:d*
dtype02
batchnorm/ReadVariableOp_2?
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:d2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????d2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*'
_output_shapes
:?????????d2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????d::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
?
K
/__inference_max_pooling2d_1_layer_call_fn_25709

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_257032
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
H__inference_binary_conv2d_layer_call_and_return_conditional_losses_25689

inputs
readvariableop_resource
identity??ReadVariableOp?
ReadVariableOpReadVariableOpreadvariableop_resource*&
_output_shapes
:  *
dtype02
ReadVariableOp[
	truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
	truediv/yz
truedivRealDivReadVariableOp:value:0truediv/y:output:0*
T0*&
_output_shapes
:  2	
truedivS
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
mul/x_
mulMulmul/x:output:0truediv:z:0*
T0*&
_output_shapes
:  2
mulS
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
add/y]
addAddV2mul:z:0add/y:output:0*
T0*&
_output_shapes
:  2
addw
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value/Minimum/y?
clip_by_value/MinimumMinimumadd:z:0 clip_by_value/Minimum/y:output:0*
T0*&
_output_shapes
:  2
clip_by_value/Minimumg
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value/y?
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*&
_output_shapes
:  2
clip_by_value[
RoundRoundclip_by_value:z:0*
T0*&
_output_shapes
:  2
Round`
subSub	Round:y:0clip_by_value:z:0*
T0*&
_output_shapes
:  2
subf
StopGradientStopGradientsub:z:0*
T0*&
_output_shapes
:  2
StopGradientr
add_1AddV2clip_by_value:z:0StopGradient:output:0*
T0*&
_output_shapes
:  2
add_1W
mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @2	
mul_1/xc
mul_1Mulmul_1/x:output:0	add_1:z:0*
T0*&
_output_shapes
:  2
mul_1W
sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2	
sub_1/yc
sub_1Sub	mul_1:z:0sub_1/y:output:0*
T0*&
_output_shapes
:  2
sub_1W
mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2	
mul_2/xc
mul_2Mulmul_2/x:output:0	sub_1:z:0*
T0*&
_output_shapes
:  2
mul_2?
convolutionConv2Dinputs	mul_2:z:0*
T0*A
_output_shapes/
-:+??????????????????????????? *
paddingSAME*
strides
2
convolution?
IdentityIdentityconvolution:output:0^ReadVariableOp*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*D
_input_shapes3
1:+??????????????????????????? :2 
ReadVariableOpReadVariableOp:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?<
?
E__inference_sequential_layer_call_and_return_conditional_losses_26617

inputs
conv1_26562
	bn1_26566
	bn1_26568
	bn1_26570
	bn1_26572
binary_conv2d_26576
batch_normalization_26580
batch_normalization_26582
batch_normalization_26584
batch_normalization_26586
dense5_26591
	bn5_26594
	bn5_26596
	bn5_26598
	bn5_26600
dense6_26604
	bn6_26607
	bn6_26609
	bn6_26611
	bn6_26613
identity??+batch_normalization/StatefulPartitionedCall?%binary_conv2d/StatefulPartitionedCall?bn1/StatefulPartitionedCall?bn5/StatefulPartitionedCall?bn6/StatefulPartitionedCall?conv1/StatefulPartitionedCall?dense5/StatefulPartitionedCall?dense6/StatefulPartitionedCall?
conv1/StatefulPartitionedCallStatefulPartitionedCallinputsconv1_26562*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@ *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_conv1_layer_call_and_return_conditional_losses_255372
conv1/StatefulPartitionedCall?
max_pooling2d/PartitionedCallPartitionedCall&conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????   * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_255512
max_pooling2d/PartitionedCall?
bn1/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0	bn1_26566	bn1_26568	bn1_26570	bn1_26572*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????   *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *G
fBR@
>__inference_bn1_layer_call_and_return_conditional_losses_261202
bn1/StatefulPartitionedCall?
act1/PartitionedCallPartitionedCall$bn1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????   * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *H
fCRA
?__inference_act1_layer_call_and_return_conditional_losses_261942
act1/PartitionedCall?
%binary_conv2d/StatefulPartitionedCallStatefulPartitionedCallact1/PartitionedCall:output:0binary_conv2d_26576*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????   *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_binary_conv2d_layer_call_and_return_conditional_losses_256892'
%binary_conv2d/StatefulPartitionedCall?
max_pooling2d_1/PartitionedCallPartitionedCall.binary_conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_257032!
max_pooling2d_1/PartitionedCall?
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_1/PartitionedCall:output:0batch_normalization_26580batch_normalization_26582batch_normalization_26584batch_normalization_26586*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_262252-
+batch_normalization/StatefulPartitionedCall?
activation/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_activation_layer_call_and_return_conditional_losses_262992
activation/PartitionedCall?
flatten/PartitionedCallPartitionedCall#activation/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_263132
flatten/PartitionedCall?
dense5/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense5_26591*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_dense5_layer_call_and_return_conditional_losses_263482 
dense5/StatefulPartitionedCall?
bn5/StatefulPartitionedCallStatefulPartitionedCall'dense5/StatefulPartitionedCall:output:0	bn5_26594	bn5_26596	bn5_26598	bn5_26600*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *G
fBR@
>__inference_bn5_layer_call_and_return_conditional_losses_259092
bn5/StatefulPartitionedCall?
act5/PartitionedCallPartitionedCall$bn5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *H
fCRA
?__inference_act5_layer_call_and_return_conditional_losses_264152
act5/PartitionedCall?
dense6/StatefulPartitionedCallStatefulPartitionedCallact5/PartitionedCall:output:0dense6_26604*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_dense6_layer_call_and_return_conditional_losses_264502 
dense6/StatefulPartitionedCall?
bn6/StatefulPartitionedCallStatefulPartitionedCall'dense6/StatefulPartitionedCall:output:0	bn6_26607	bn6_26609	bn6_26611	bn6_26613*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *G
fBR@
>__inference_bn6_layer_call_and_return_conditional_losses_260492
bn6/StatefulPartitionedCall?
IdentityIdentity$bn6/StatefulPartitionedCall:output:0,^batch_normalization/StatefulPartitionedCall&^binary_conv2d/StatefulPartitionedCall^bn1/StatefulPartitionedCall^bn5/StatefulPartitionedCall^bn6/StatefulPartitionedCall^conv1/StatefulPartitionedCall^dense5/StatefulPartitionedCall^dense6/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????d2

Identity"
identityIdentity:output:0*~
_input_shapesm
k:?????????@@::::::::::::::::::::2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2N
%binary_conv2d/StatefulPartitionedCall%binary_conv2d/StatefulPartitionedCall2:
bn1/StatefulPartitionedCallbn1/StatefulPartitionedCall2:
bn5/StatefulPartitionedCallbn5/StatefulPartitionedCall2:
bn6/StatefulPartitionedCallbn6/StatefulPartitionedCall2>
conv1/StatefulPartitionedCallconv1/StatefulPartitionedCall2@
dense5/StatefulPartitionedCalldense5/StatefulPartitionedCall2@
dense6/StatefulPartitionedCalldense6/StatefulPartitionedCall:W S
/
_output_shapes
:?????????@@
 
_user_specified_nameinputs
??
?
E__inference_sequential_layer_call_and_return_conditional_losses_27270

inputs!
conv1_readvariableop_resource
bn1_readvariableop_resource!
bn1_readvariableop_1_resource0
,bn1_fusedbatchnormv3_readvariableop_resource2
.bn1_fusedbatchnormv3_readvariableop_1_resource)
%binary_conv2d_readvariableop_resource/
+batch_normalization_readvariableop_resource1
-batch_normalization_readvariableop_1_resource@
<batch_normalization_fusedbatchnormv3_readvariableop_resourceB
>batch_normalization_fusedbatchnormv3_readvariableop_1_resource"
dense5_readvariableop_resource)
%bn5_batchnorm_readvariableop_resource-
)bn5_batchnorm_mul_readvariableop_resource+
'bn5_batchnorm_readvariableop_1_resource+
'bn5_batchnorm_readvariableop_2_resource"
dense6_readvariableop_resource)
%bn6_batchnorm_readvariableop_resource-
)bn6_batchnorm_mul_readvariableop_resource+
'bn6_batchnorm_readvariableop_1_resource+
'bn6_batchnorm_readvariableop_2_resource
identity??3batch_normalization/FusedBatchNormV3/ReadVariableOp?5batch_normalization/FusedBatchNormV3/ReadVariableOp_1?"batch_normalization/ReadVariableOp?$batch_normalization/ReadVariableOp_1?binary_conv2d/ReadVariableOp?#bn1/FusedBatchNormV3/ReadVariableOp?%bn1/FusedBatchNormV3/ReadVariableOp_1?bn1/ReadVariableOp?bn1/ReadVariableOp_1?bn5/batchnorm/ReadVariableOp?bn5/batchnorm/ReadVariableOp_1?bn5/batchnorm/ReadVariableOp_2? bn5/batchnorm/mul/ReadVariableOp?bn6/batchnorm/ReadVariableOp?bn6/batchnorm/ReadVariableOp_1?bn6/batchnorm/ReadVariableOp_2? bn6/batchnorm/mul/ReadVariableOp?conv1/ReadVariableOp?dense5/ReadVariableOp?dense6/ReadVariableOp?
conv1/ReadVariableOpReadVariableOpconv1_readvariableop_resource*&
_output_shapes
: *
dtype02
conv1/ReadVariableOpg
conv1/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
conv1/truediv/y?
conv1/truedivRealDivconv1/ReadVariableOp:value:0conv1/truediv/y:output:0*
T0*&
_output_shapes
: 2
conv1/truediv_
conv1/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
conv1/mul/xw
	conv1/mulMulconv1/mul/x:output:0conv1/truediv:z:0*
T0*&
_output_shapes
: 2
	conv1/mul_
conv1/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
conv1/add/yu
	conv1/addAddV2conv1/mul:z:0conv1/add/y:output:0*
T0*&
_output_shapes
: 2
	conv1/add?
conv1/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
conv1/clip_by_value/Minimum/y?
conv1/clip_by_value/MinimumMinimumconv1/add:z:0&conv1/clip_by_value/Minimum/y:output:0*
T0*&
_output_shapes
: 2
conv1/clip_by_value/Minimums
conv1/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
conv1/clip_by_value/y?
conv1/clip_by_valueMaximumconv1/clip_by_value/Minimum:z:0conv1/clip_by_value/y:output:0*
T0*&
_output_shapes
: 2
conv1/clip_by_valuem
conv1/RoundRoundconv1/clip_by_value:z:0*
T0*&
_output_shapes
: 2
conv1/Roundx
	conv1/subSubconv1/Round:y:0conv1/clip_by_value:z:0*
T0*&
_output_shapes
: 2
	conv1/subx
conv1/StopGradientStopGradientconv1/sub:z:0*
T0*&
_output_shapes
: 2
conv1/StopGradient?
conv1/add_1AddV2conv1/clip_by_value:z:0conv1/StopGradient:output:0*
T0*&
_output_shapes
: 2
conv1/add_1c
conv1/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
conv1/mul_1/x{
conv1/mul_1Mulconv1/mul_1/x:output:0conv1/add_1:z:0*
T0*&
_output_shapes
: 2
conv1/mul_1c
conv1/sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
conv1/sub_1/y{
conv1/sub_1Subconv1/mul_1:z:0conv1/sub_1/y:output:0*
T0*&
_output_shapes
: 2
conv1/sub_1c
conv1/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
conv1/mul_2/x{
conv1/mul_2Mulconv1/mul_2/x:output:0conv1/sub_1:z:0*
T0*&
_output_shapes
: 2
conv1/mul_2?
conv1/convolutionConv2Dinputsconv1/mul_2:z:0*
T0*/
_output_shapes
:?????????@@ *
paddingSAME*
strides
2
conv1/convolution?
max_pooling2d/MaxPoolMaxPoolconv1/convolution:output:0*/
_output_shapes
:?????????   *
ksize
*
paddingVALID*
strides
2
max_pooling2d/MaxPool?
bn1/ReadVariableOpReadVariableOpbn1_readvariableop_resource*
_output_shapes
: *
dtype02
bn1/ReadVariableOp?
bn1/ReadVariableOp_1ReadVariableOpbn1_readvariableop_1_resource*
_output_shapes
: *
dtype02
bn1/ReadVariableOp_1?
#bn1/FusedBatchNormV3/ReadVariableOpReadVariableOp,bn1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02%
#bn1/FusedBatchNormV3/ReadVariableOp?
%bn1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp.bn1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02'
%bn1/FusedBatchNormV3/ReadVariableOp_1?
bn1/FusedBatchNormV3FusedBatchNormV3max_pooling2d/MaxPool:output:0bn1/ReadVariableOp:value:0bn1/ReadVariableOp_1:value:0+bn1/FusedBatchNormV3/ReadVariableOp:value:0-bn1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????   : : : : :*
data_formatNCHW*
epsilon%??'7*
is_training( 2
bn1/FusedBatchNormV3]

act1/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2

act1/mul/x?
act1/mulMulact1/mul/x:output:0bn1/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????   2

act1/mul]

act1/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2

act1/add/yz
act1/addAddV2act1/mul:z:0act1/add/y:output:0*
T0*/
_output_shapes
:?????????   2

act1/add?
act1/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
act1/clip_by_value/Minimum/y?
act1/clip_by_value/MinimumMinimumact1/add:z:0%act1/clip_by_value/Minimum/y:output:0*
T0*/
_output_shapes
:?????????   2
act1/clip_by_value/Minimumq
act1/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
act1/clip_by_value/y?
act1/clip_by_valueMaximumact1/clip_by_value/Minimum:z:0act1/clip_by_value/y:output:0*
T0*/
_output_shapes
:?????????   2
act1/clip_by_values

act1/RoundRoundact1/clip_by_value:z:0*
T0*/
_output_shapes
:?????????   2

act1/Round}
act1/subSubact1/Round:y:0act1/clip_by_value:z:0*
T0*/
_output_shapes
:?????????   2

act1/sub~
act1/StopGradientStopGradientact1/sub:z:0*
T0*/
_output_shapes
:?????????   2
act1/StopGradient?

act1/add_1AddV2act1/clip_by_value:z:0act1/StopGradient:output:0*
T0*/
_output_shapes
:?????????   2

act1/add_1a
act1/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
act1/mul_1/x?

act1/mul_1Mulact1/mul_1/x:output:0act1/add_1:z:0*
T0*/
_output_shapes
:?????????   2

act1/mul_1a
act1/sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
act1/sub_1/y?

act1/sub_1Subact1/mul_1:z:0act1/sub_1/y:output:0*
T0*/
_output_shapes
:?????????   2

act1/sub_1?
binary_conv2d/ReadVariableOpReadVariableOp%binary_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02
binary_conv2d/ReadVariableOpw
binary_conv2d/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
binary_conv2d/truediv/y?
binary_conv2d/truedivRealDiv$binary_conv2d/ReadVariableOp:value:0 binary_conv2d/truediv/y:output:0*
T0*&
_output_shapes
:  2
binary_conv2d/truedivo
binary_conv2d/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
binary_conv2d/mul/x?
binary_conv2d/mulMulbinary_conv2d/mul/x:output:0binary_conv2d/truediv:z:0*
T0*&
_output_shapes
:  2
binary_conv2d/mulo
binary_conv2d/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
binary_conv2d/add/y?
binary_conv2d/addAddV2binary_conv2d/mul:z:0binary_conv2d/add/y:output:0*
T0*&
_output_shapes
:  2
binary_conv2d/add?
%binary_conv2d/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2'
%binary_conv2d/clip_by_value/Minimum/y?
#binary_conv2d/clip_by_value/MinimumMinimumbinary_conv2d/add:z:0.binary_conv2d/clip_by_value/Minimum/y:output:0*
T0*&
_output_shapes
:  2%
#binary_conv2d/clip_by_value/Minimum?
binary_conv2d/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
binary_conv2d/clip_by_value/y?
binary_conv2d/clip_by_valueMaximum'binary_conv2d/clip_by_value/Minimum:z:0&binary_conv2d/clip_by_value/y:output:0*
T0*&
_output_shapes
:  2
binary_conv2d/clip_by_value?
binary_conv2d/RoundRoundbinary_conv2d/clip_by_value:z:0*
T0*&
_output_shapes
:  2
binary_conv2d/Round?
binary_conv2d/subSubbinary_conv2d/Round:y:0binary_conv2d/clip_by_value:z:0*
T0*&
_output_shapes
:  2
binary_conv2d/sub?
binary_conv2d/StopGradientStopGradientbinary_conv2d/sub:z:0*
T0*&
_output_shapes
:  2
binary_conv2d/StopGradient?
binary_conv2d/add_1AddV2binary_conv2d/clip_by_value:z:0#binary_conv2d/StopGradient:output:0*
T0*&
_output_shapes
:  2
binary_conv2d/add_1s
binary_conv2d/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
binary_conv2d/mul_1/x?
binary_conv2d/mul_1Mulbinary_conv2d/mul_1/x:output:0binary_conv2d/add_1:z:0*
T0*&
_output_shapes
:  2
binary_conv2d/mul_1s
binary_conv2d/sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
binary_conv2d/sub_1/y?
binary_conv2d/sub_1Subbinary_conv2d/mul_1:z:0binary_conv2d/sub_1/y:output:0*
T0*&
_output_shapes
:  2
binary_conv2d/sub_1s
binary_conv2d/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
binary_conv2d/mul_2/x?
binary_conv2d/mul_2Mulbinary_conv2d/mul_2/x:output:0binary_conv2d/sub_1:z:0*
T0*&
_output_shapes
:  2
binary_conv2d/mul_2?
binary_conv2d/convolutionConv2Dact1/sub_1:z:0binary_conv2d/mul_2:z:0*
T0*/
_output_shapes
:?????????   *
paddingSAME*
strides
2
binary_conv2d/convolution?
max_pooling2d_1/MaxPoolMaxPool"binary_conv2d/convolution:output:0*/
_output_shapes
:????????? *
ksize
*
paddingVALID*
strides
2
max_pooling2d_1/MaxPool?
"batch_normalization/ReadVariableOpReadVariableOp+batch_normalization_readvariableop_resource*
_output_shapes
:*
dtype02$
"batch_normalization/ReadVariableOp?
$batch_normalization/ReadVariableOp_1ReadVariableOp-batch_normalization_readvariableop_1_resource*
_output_shapes
:*
dtype02&
$batch_normalization/ReadVariableOp_1?
3batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype025
3batch_normalization/FusedBatchNormV3/ReadVariableOp?
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype027
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1?
$batch_normalization/FusedBatchNormV3FusedBatchNormV3 max_pooling2d_1/MaxPool:output:0*batch_normalization/ReadVariableOp:value:0,batch_normalization/ReadVariableOp_1:value:0;batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0=batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:????????? :::::*
data_formatNCHW*
epsilon%??'7*
is_training( 2&
$batch_normalization/FusedBatchNormV3i
activation/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
activation/mul/x?
activation/mulMulactivation/mul/x:output:0(batch_normalization/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:????????? 2
activation/muli
activation/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
activation/add/y?
activation/addAddV2activation/mul:z:0activation/add/y:output:0*
T0*/
_output_shapes
:????????? 2
activation/add?
"activation/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2$
"activation/clip_by_value/Minimum/y?
 activation/clip_by_value/MinimumMinimumactivation/add:z:0+activation/clip_by_value/Minimum/y:output:0*
T0*/
_output_shapes
:????????? 2"
 activation/clip_by_value/Minimum}
activation/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
activation/clip_by_value/y?
activation/clip_by_valueMaximum$activation/clip_by_value/Minimum:z:0#activation/clip_by_value/y:output:0*
T0*/
_output_shapes
:????????? 2
activation/clip_by_value?
activation/RoundRoundactivation/clip_by_value:z:0*
T0*/
_output_shapes
:????????? 2
activation/Round?
activation/subSubactivation/Round:y:0activation/clip_by_value:z:0*
T0*/
_output_shapes
:????????? 2
activation/sub?
activation/StopGradientStopGradientactivation/sub:z:0*
T0*/
_output_shapes
:????????? 2
activation/StopGradient?
activation/add_1AddV2activation/clip_by_value:z:0 activation/StopGradient:output:0*
T0*/
_output_shapes
:????????? 2
activation/add_1m
activation/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
activation/mul_1/x?
activation/mul_1Mulactivation/mul_1/x:output:0activation/add_1:z:0*
T0*/
_output_shapes
:????????? 2
activation/mul_1m
activation/sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
activation/sub_1/y?
activation/sub_1Subactivation/mul_1:z:0activation/sub_1/y:output:0*
T0*/
_output_shapes
:????????? 2
activation/sub_1o
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????    2
flatten/Const?
flatten/ReshapeReshapeactivation/sub_1:z:0flatten/Const:output:0*
T0*(
_output_shapes
:??????????@2
flatten/Reshape?
dense5/ReadVariableOpReadVariableOpdense5_readvariableop_resource* 
_output_shapes
:
?@?*
dtype02
dense5/ReadVariableOpi
dense5/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dense5/truediv/y?
dense5/truedivRealDivdense5/ReadVariableOp:value:0dense5/truediv/y:output:0*
T0* 
_output_shapes
:
?@?2
dense5/truediva
dense5/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dense5/mul/xu

dense5/mulMuldense5/mul/x:output:0dense5/truediv:z:0*
T0* 
_output_shapes
:
?@?2

dense5/mula
dense5/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dense5/add/ys

dense5/addAddV2dense5/mul:z:0dense5/add/y:output:0*
T0* 
_output_shapes
:
?@?2

dense5/add?
dense5/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2 
dense5/clip_by_value/Minimum/y?
dense5/clip_by_value/MinimumMinimumdense5/add:z:0'dense5/clip_by_value/Minimum/y:output:0*
T0* 
_output_shapes
:
?@?2
dense5/clip_by_value/Minimumu
dense5/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
dense5/clip_by_value/y?
dense5/clip_by_valueMaximum dense5/clip_by_value/Minimum:z:0dense5/clip_by_value/y:output:0*
T0* 
_output_shapes
:
?@?2
dense5/clip_by_valuej
dense5/RoundRounddense5/clip_by_value:z:0*
T0* 
_output_shapes
:
?@?2
dense5/Roundv

dense5/subSubdense5/Round:y:0dense5/clip_by_value:z:0*
T0* 
_output_shapes
:
?@?2

dense5/subu
dense5/StopGradientStopGradientdense5/sub:z:0*
T0* 
_output_shapes
:
?@?2
dense5/StopGradient?
dense5/add_1AddV2dense5/clip_by_value:z:0dense5/StopGradient:output:0*
T0* 
_output_shapes
:
?@?2
dense5/add_1e
dense5/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dense5/mul_1/xy
dense5/mul_1Muldense5/mul_1/x:output:0dense5/add_1:z:0*
T0* 
_output_shapes
:
?@?2
dense5/mul_1e
dense5/sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dense5/sub_1/yy
dense5/sub_1Subdense5/mul_1:z:0dense5/sub_1/y:output:0*
T0* 
_output_shapes
:
?@?2
dense5/sub_1e
dense5/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dense5/mul_2/xy
dense5/mul_2Muldense5/mul_2/x:output:0dense5/sub_1:z:0*
T0* 
_output_shapes
:
?@?2
dense5/mul_2?
dense5/MatMulMatMulflatten/Reshape:output:0dense5/mul_2:z:0*
T0*(
_output_shapes
:??????????2
dense5/MatMul?
bn5/batchnorm/ReadVariableOpReadVariableOp%bn5_batchnorm_readvariableop_resource*
_output_shapes	
:?*
dtype02
bn5/batchnorm/ReadVariableOpo
bn5/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *?7?52
bn5/batchnorm/add/y?
bn5/batchnorm/addAddV2$bn5/batchnorm/ReadVariableOp:value:0bn5/batchnorm/add/y:output:0*
T0*
_output_shapes	
:?2
bn5/batchnorm/addp
bn5/batchnorm/RsqrtRsqrtbn5/batchnorm/add:z:0*
T0*
_output_shapes	
:?2
bn5/batchnorm/Rsqrt?
 bn5/batchnorm/mul/ReadVariableOpReadVariableOp)bn5_batchnorm_mul_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 bn5/batchnorm/mul/ReadVariableOp?
bn5/batchnorm/mulMulbn5/batchnorm/Rsqrt:y:0(bn5/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:?2
bn5/batchnorm/mul?
bn5/batchnorm/mul_1Muldense5/MatMul:product:0bn5/batchnorm/mul:z:0*
T0*(
_output_shapes
:??????????2
bn5/batchnorm/mul_1?
bn5/batchnorm/ReadVariableOp_1ReadVariableOp'bn5_batchnorm_readvariableop_1_resource*
_output_shapes	
:?*
dtype02 
bn5/batchnorm/ReadVariableOp_1?
bn5/batchnorm/mul_2Mul&bn5/batchnorm/ReadVariableOp_1:value:0bn5/batchnorm/mul:z:0*
T0*
_output_shapes	
:?2
bn5/batchnorm/mul_2?
bn5/batchnorm/ReadVariableOp_2ReadVariableOp'bn5_batchnorm_readvariableop_2_resource*
_output_shapes	
:?*
dtype02 
bn5/batchnorm/ReadVariableOp_2?
bn5/batchnorm/subSub&bn5/batchnorm/ReadVariableOp_2:value:0bn5/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?2
bn5/batchnorm/sub?
bn5/batchnorm/add_1AddV2bn5/batchnorm/mul_1:z:0bn5/batchnorm/sub:z:0*
T0*(
_output_shapes
:??????????2
bn5/batchnorm/add_1]

act5/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2

act5/mul/x|
act5/mulMulact5/mul/x:output:0bn5/batchnorm/add_1:z:0*
T0*(
_output_shapes
:??????????2

act5/mul]

act5/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2

act5/add/ys
act5/addAddV2act5/mul:z:0act5/add/y:output:0*
T0*(
_output_shapes
:??????????2

act5/add?
act5/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
act5/clip_by_value/Minimum/y?
act5/clip_by_value/MinimumMinimumact5/add:z:0%act5/clip_by_value/Minimum/y:output:0*
T0*(
_output_shapes
:??????????2
act5/clip_by_value/Minimumq
act5/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
act5/clip_by_value/y?
act5/clip_by_valueMaximumact5/clip_by_value/Minimum:z:0act5/clip_by_value/y:output:0*
T0*(
_output_shapes
:??????????2
act5/clip_by_valuel

act5/RoundRoundact5/clip_by_value:z:0*
T0*(
_output_shapes
:??????????2

act5/Roundv
act5/subSubact5/Round:y:0act5/clip_by_value:z:0*
T0*(
_output_shapes
:??????????2

act5/subw
act5/StopGradientStopGradientact5/sub:z:0*
T0*(
_output_shapes
:??????????2
act5/StopGradient?

act5/add_1AddV2act5/clip_by_value:z:0act5/StopGradient:output:0*
T0*(
_output_shapes
:??????????2

act5/add_1a
act5/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
act5/mul_1/xy

act5/mul_1Mulact5/mul_1/x:output:0act5/add_1:z:0*
T0*(
_output_shapes
:??????????2

act5/mul_1a
act5/sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
act5/sub_1/yy

act5/sub_1Subact5/mul_1:z:0act5/sub_1/y:output:0*
T0*(
_output_shapes
:??????????2

act5/sub_1?
dense6/ReadVariableOpReadVariableOpdense6_readvariableop_resource*
_output_shapes
:	?d*
dtype02
dense6/ReadVariableOpi
dense6/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dense6/truediv/y?
dense6/truedivRealDivdense6/ReadVariableOp:value:0dense6/truediv/y:output:0*
T0*
_output_shapes
:	?d2
dense6/truediva
dense6/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dense6/mul/xt

dense6/mulMuldense6/mul/x:output:0dense6/truediv:z:0*
T0*
_output_shapes
:	?d2

dense6/mula
dense6/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dense6/add/yr

dense6/addAddV2dense6/mul:z:0dense6/add/y:output:0*
T0*
_output_shapes
:	?d2

dense6/add?
dense6/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2 
dense6/clip_by_value/Minimum/y?
dense6/clip_by_value/MinimumMinimumdense6/add:z:0'dense6/clip_by_value/Minimum/y:output:0*
T0*
_output_shapes
:	?d2
dense6/clip_by_value/Minimumu
dense6/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
dense6/clip_by_value/y?
dense6/clip_by_valueMaximum dense6/clip_by_value/Minimum:z:0dense6/clip_by_value/y:output:0*
T0*
_output_shapes
:	?d2
dense6/clip_by_valuei
dense6/RoundRounddense6/clip_by_value:z:0*
T0*
_output_shapes
:	?d2
dense6/Roundu

dense6/subSubdense6/Round:y:0dense6/clip_by_value:z:0*
T0*
_output_shapes
:	?d2

dense6/subt
dense6/StopGradientStopGradientdense6/sub:z:0*
T0*
_output_shapes
:	?d2
dense6/StopGradient?
dense6/add_1AddV2dense6/clip_by_value:z:0dense6/StopGradient:output:0*
T0*
_output_shapes
:	?d2
dense6/add_1e
dense6/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dense6/mul_1/xx
dense6/mul_1Muldense6/mul_1/x:output:0dense6/add_1:z:0*
T0*
_output_shapes
:	?d2
dense6/mul_1e
dense6/sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dense6/sub_1/yx
dense6/sub_1Subdense6/mul_1:z:0dense6/sub_1/y:output:0*
T0*
_output_shapes
:	?d2
dense6/sub_1e
dense6/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dense6/mul_2/xx
dense6/mul_2Muldense6/mul_2/x:output:0dense6/sub_1:z:0*
T0*
_output_shapes
:	?d2
dense6/mul_2|
dense6/MatMulMatMulact5/sub_1:z:0dense6/mul_2:z:0*
T0*'
_output_shapes
:?????????d2
dense6/MatMul?
bn6/batchnorm/ReadVariableOpReadVariableOp%bn6_batchnorm_readvariableop_resource*
_output_shapes
:d*
dtype02
bn6/batchnorm/ReadVariableOpo
bn6/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *?7?52
bn6/batchnorm/add/y?
bn6/batchnorm/addAddV2$bn6/batchnorm/ReadVariableOp:value:0bn6/batchnorm/add/y:output:0*
T0*
_output_shapes
:d2
bn6/batchnorm/addo
bn6/batchnorm/RsqrtRsqrtbn6/batchnorm/add:z:0*
T0*
_output_shapes
:d2
bn6/batchnorm/Rsqrt?
 bn6/batchnorm/mul/ReadVariableOpReadVariableOp)bn6_batchnorm_mul_readvariableop_resource*
_output_shapes
:d*
dtype02"
 bn6/batchnorm/mul/ReadVariableOp?
bn6/batchnorm/mulMulbn6/batchnorm/Rsqrt:y:0(bn6/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:d2
bn6/batchnorm/mul?
bn6/batchnorm/mul_1Muldense6/MatMul:product:0bn6/batchnorm/mul:z:0*
T0*'
_output_shapes
:?????????d2
bn6/batchnorm/mul_1?
bn6/batchnorm/ReadVariableOp_1ReadVariableOp'bn6_batchnorm_readvariableop_1_resource*
_output_shapes
:d*
dtype02 
bn6/batchnorm/ReadVariableOp_1?
bn6/batchnorm/mul_2Mul&bn6/batchnorm/ReadVariableOp_1:value:0bn6/batchnorm/mul:z:0*
T0*
_output_shapes
:d2
bn6/batchnorm/mul_2?
bn6/batchnorm/ReadVariableOp_2ReadVariableOp'bn6_batchnorm_readvariableop_2_resource*
_output_shapes
:d*
dtype02 
bn6/batchnorm/ReadVariableOp_2?
bn6/batchnorm/subSub&bn6/batchnorm/ReadVariableOp_2:value:0bn6/batchnorm/mul_2:z:0*
T0*
_output_shapes
:d2
bn6/batchnorm/sub?
bn6/batchnorm/add_1AddV2bn6/batchnorm/mul_1:z:0bn6/batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????d2
bn6/batchnorm/add_1?
IdentityIdentitybn6/batchnorm/add_1:z:04^batch_normalization/FusedBatchNormV3/ReadVariableOp6^batch_normalization/FusedBatchNormV3/ReadVariableOp_1#^batch_normalization/ReadVariableOp%^batch_normalization/ReadVariableOp_1^binary_conv2d/ReadVariableOp$^bn1/FusedBatchNormV3/ReadVariableOp&^bn1/FusedBatchNormV3/ReadVariableOp_1^bn1/ReadVariableOp^bn1/ReadVariableOp_1^bn5/batchnorm/ReadVariableOp^bn5/batchnorm/ReadVariableOp_1^bn5/batchnorm/ReadVariableOp_2!^bn5/batchnorm/mul/ReadVariableOp^bn6/batchnorm/ReadVariableOp^bn6/batchnorm/ReadVariableOp_1^bn6/batchnorm/ReadVariableOp_2!^bn6/batchnorm/mul/ReadVariableOp^conv1/ReadVariableOp^dense5/ReadVariableOp^dense6/ReadVariableOp*
T0*'
_output_shapes
:?????????d2

Identity"
identityIdentity:output:0*~
_input_shapesm
k:?????????@@::::::::::::::::::::2j
3batch_normalization/FusedBatchNormV3/ReadVariableOp3batch_normalization/FusedBatchNormV3/ReadVariableOp2n
5batch_normalization/FusedBatchNormV3/ReadVariableOp_15batch_normalization/FusedBatchNormV3/ReadVariableOp_12H
"batch_normalization/ReadVariableOp"batch_normalization/ReadVariableOp2L
$batch_normalization/ReadVariableOp_1$batch_normalization/ReadVariableOp_12<
binary_conv2d/ReadVariableOpbinary_conv2d/ReadVariableOp2J
#bn1/FusedBatchNormV3/ReadVariableOp#bn1/FusedBatchNormV3/ReadVariableOp2N
%bn1/FusedBatchNormV3/ReadVariableOp_1%bn1/FusedBatchNormV3/ReadVariableOp_12(
bn1/ReadVariableOpbn1/ReadVariableOp2,
bn1/ReadVariableOp_1bn1/ReadVariableOp_12<
bn5/batchnorm/ReadVariableOpbn5/batchnorm/ReadVariableOp2@
bn5/batchnorm/ReadVariableOp_1bn5/batchnorm/ReadVariableOp_12@
bn5/batchnorm/ReadVariableOp_2bn5/batchnorm/ReadVariableOp_22D
 bn5/batchnorm/mul/ReadVariableOp bn5/batchnorm/mul/ReadVariableOp2<
bn6/batchnorm/ReadVariableOpbn6/batchnorm/ReadVariableOp2@
bn6/batchnorm/ReadVariableOp_1bn6/batchnorm/ReadVariableOp_12@
bn6/batchnorm/ReadVariableOp_2bn6/batchnorm/ReadVariableOp_22D
 bn6/batchnorm/mul/ReadVariableOp bn6/batchnorm/mul/ReadVariableOp2,
conv1/ReadVariableOpconv1/ReadVariableOp2.
dense5/ReadVariableOpdense5/ReadVariableOp2.
dense6/ReadVariableOpdense6/ReadVariableOp:W S
/
_output_shapes
:?????????@@
 
_user_specified_nameinputs
?
?
*__inference_sequential_layer_call_fn_27315

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18* 
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d*.
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_266172
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????d2

Identity"
identityIdentity:output:0*~
_input_shapesm
k:?????????@@::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????@@
 
_user_specified_nameinputs
?
?
>__inference_bn6_layer_call_and_return_conditional_losses_26082

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identity??batchnorm/ReadVariableOp?batchnorm/ReadVariableOp_1?batchnorm/ReadVariableOp_2?batchnorm/mul/ReadVariableOp?
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:d*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *?7?52
batchnorm/add/y?
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:d2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:d2
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:d*
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:d2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:?????????d2
batchnorm/mul_1?
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:d*
dtype02
batchnorm/ReadVariableOp_1?
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:d2
batchnorm/mul_2?
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:d*
dtype02
batchnorm/ReadVariableOp_2?
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:d2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????d2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*'
_output_shapes
:?????????d2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????d::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
?
?
*__inference_sequential_layer_call_fn_26763
conv1_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallconv1_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18* 
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d*6
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_267202
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????d2

Identity"
identityIdentity:output:0*~
_input_shapesm
k:?????????@@::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:\ X
/
_output_shapes
:?????????@@
%
_user_specified_nameconv1_input"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
K
conv1_input<
serving_default_conv1_input:0?????????@@7
bn60
StatefulPartitionedCall:0?????????dtensorflow/serving/predict:??
?r
layer_with_weights-0
layer-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer-5
layer_with_weights-3
layer-6
layer-7
	layer-8

layer_with_weights-4

layer-9
layer_with_weights-5
layer-10
layer-11
layer_with_weights-6
layer-12
layer_with_weights-7
layer-13
	optimizer
regularization_losses
trainable_variables
	variables
	keras_api

signatures
?_default_save_signature
+?&call_and_return_all_conditional_losses
?__call__"?m
_tf_keras_sequential?m{"class_name": "Sequential", "name": "sequential", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "sequential", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 64, 64, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv1_input"}}, {"class_name": "BinaryConv2D", "config": {"name": "conv1", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 64, 64, 3]}, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "RandomUniform", "config": {"minval": -1.0, "maxval": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": {"class_name": "Clip", "config": {"min_value": -1.0, "max_value": 1.0}}, "bias_constraint": null, "H": 1.0, "kernel_lr_multiplier": 14.491376876831055, "bias_lr_multiplier": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "BatchNormalization", "config": {"name": "bn1", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.9, "epsilon": 1e-06, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Activation", "config": {"name": "act1", "trainable": true, "dtype": "float32", "activation": "binary_tanh"}}, {"class_name": "BinaryConv2D", "config": {"name": "binary_conv2d", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "RandomUniform", "config": {"minval": -1.0, "maxval": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": {"class_name": "Clip", "config": {"min_value": -1.0, "max_value": 1.0}}, "bias_constraint": null, "H": 1.0, "kernel_lr_multiplier": 19.595918655395508, "bias_lr_multiplier": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_1", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.9, "epsilon": 1e-06, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation", "trainable": true, "dtype": "float32", "activation": "binary_tanh"}}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "BinaryDense", "config": {"name": "dense5", "trainable": true, "dtype": "float32", "units": 128, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "RandomUniform", "config": {"minval": -1.0, "maxval": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": {"class_name": "Clip", "config": {"min_value": -1.0, "max_value": 1.0}}, "bias_constraint": null, "H": 1.0, "kernel_lr_multiplier": 74.47594451904297, "bias_lr_multiplier": null}}, {"class_name": "BatchNormalization", "config": {"name": "bn5", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.9, "epsilon": 1e-06, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Activation", "config": {"name": "act5", "trainable": true, "dtype": "float32", "activation": "binary_tanh"}}, {"class_name": "BinaryDense", "config": {"name": "dense6", "trainable": true, "dtype": "float32", "units": 100, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "RandomUniform", "config": {"minval": -1.0, "maxval": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": {"class_name": "Clip", "config": {"min_value": -1.0, "max_value": 1.0}}, "bias_constraint": null, "H": 1.0, "kernel_lr_multiplier": 12.328827857971191, "bias_lr_multiplier": null}}, {"class_name": "BatchNormalization", "config": {"name": "bn6", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.9, "epsilon": 1e-06, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 3}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64, 64, 3]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 64, 64, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv1_input"}}, {"class_name": "BinaryConv2D", "config": {"name": "conv1", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 64, 64, 3]}, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "RandomUniform", "config": {"minval": -1.0, "maxval": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": {"class_name": "Clip", "config": {"min_value": -1.0, "max_value": 1.0}}, "bias_constraint": null, "H": 1.0, "kernel_lr_multiplier": 14.491376876831055, "bias_lr_multiplier": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "BatchNormalization", "config": {"name": "bn1", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.9, "epsilon": 1e-06, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Activation", "config": {"name": "act1", "trainable": true, "dtype": "float32", "activation": "binary_tanh"}}, {"class_name": "BinaryConv2D", "config": {"name": "binary_conv2d", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "RandomUniform", "config": {"minval": -1.0, "maxval": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": {"class_name": "Clip", "config": {"min_value": -1.0, "max_value": 1.0}}, "bias_constraint": null, "H": 1.0, "kernel_lr_multiplier": 19.595918655395508, "bias_lr_multiplier": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_1", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.9, "epsilon": 1e-06, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation", "trainable": true, "dtype": "float32", "activation": "binary_tanh"}}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "BinaryDense", "config": {"name": "dense5", "trainable": true, "dtype": "float32", "units": 128, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "RandomUniform", "config": {"minval": -1.0, "maxval": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": {"class_name": "Clip", "config": {"min_value": -1.0, "max_value": 1.0}}, "bias_constraint": null, "H": 1.0, "kernel_lr_multiplier": 74.47594451904297, "bias_lr_multiplier": null}}, {"class_name": "BatchNormalization", "config": {"name": "bn5", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.9, "epsilon": 1e-06, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Activation", "config": {"name": "act5", "trainable": true, "dtype": "float32", "activation": "binary_tanh"}}, {"class_name": "BinaryDense", "config": {"name": "dense6", "trainable": true, "dtype": "float32", "units": 100, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "RandomUniform", "config": {"minval": -1.0, "maxval": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": {"class_name": "Clip", "config": {"min_value": -1.0, "max_value": 1.0}}, "bias_constraint": null, "H": 1.0, "kernel_lr_multiplier": 12.328827857971191, "bias_lr_multiplier": null}}, {"class_name": "BatchNormalization", "config": {"name": "bn6", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.9, "epsilon": 1e-06, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}]}}, "training_config": {"loss": "squared_hinge", "metrics": [[{"class_name": "MeanMetricWrapper", "config": {"name": "acc", "dtype": "float32", "fn": "categorical_accuracy"}}]], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.00010023052163887769, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
?

kernel
lr_multipliers
regularization_losses
trainable_variables
	variables
	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?
{"class_name": "BinaryConv2D", "name": "conv1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 64, 64, 3]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv1", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 64, 64, 3]}, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "RandomUniform", "config": {"minval": -1.0, "maxval": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": {"class_name": "Clip", "config": {"min_value": -1.0, "max_value": 1.0}}, "bias_constraint": null, "H": 1.0, "kernel_lr_multiplier": 14.491376876831055, "bias_lr_multiplier": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 3}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64, 64, 3]}}
?
regularization_losses
trainable_variables
	variables
	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "MaxPooling2D", "name": "max_pooling2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?	
axis
	 gamma
!beta
"moving_mean
#moving_variance
$regularization_losses
%trainable_variables
&	variables
'	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "BatchNormalization", "name": "bn1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "bn1", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.9, "epsilon": 1e-06, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32, 32, 32]}}
?
(regularization_losses
)trainable_variables
*	variables
+	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Activation", "name": "act1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "act1", "trainable": true, "dtype": "float32", "activation": "binary_tanh"}}
?

,kernel
-lr_multipliers
.regularization_losses
/trainable_variables
0	variables
1	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?

_tf_keras_layer?	{"class_name": "BinaryConv2D", "name": "binary_conv2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "binary_conv2d", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "RandomUniform", "config": {"minval": -1.0, "maxval": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": {"class_name": "Clip", "config": {"min_value": -1.0, "max_value": 1.0}}, "bias_constraint": null, "H": 1.0, "kernel_lr_multiplier": 19.595918655395508, "bias_lr_multiplier": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32, 32, 32]}}
?
2regularization_losses
3trainable_variables
4	variables
5	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "MaxPooling2D", "name": "max_pooling2d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d_1", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?	
6axis
	7gamma
8beta
9moving_mean
:moving_variance
;regularization_losses
<trainable_variables
=	variables
>	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "BatchNormalization", "name": "batch_normalization", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.9, "epsilon": 1e-06, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"1": 16}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 16, 16, 32]}}
?
?regularization_losses
@trainable_variables
A	variables
B	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Activation", "name": "activation", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation", "trainable": true, "dtype": "float32", "activation": "binary_tanh"}}
?
Cregularization_losses
Dtrainable_variables
E	variables
F	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Flatten", "name": "flatten", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
?	

Gkernel
Hlr_multipliers
Iregularization_losses
Jtrainable_variables
K	variables
L	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "BinaryDense", "name": "dense5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense5", "trainable": true, "dtype": "float32", "units": 128, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "RandomUniform", "config": {"minval": -1.0, "maxval": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": {"class_name": "Clip", "config": {"min_value": -1.0, "max_value": 1.0}}, "bias_constraint": null, "H": 1.0, "kernel_lr_multiplier": 74.47594451904297, "bias_lr_multiplier": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 8192}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 8192]}}
?	
Maxis
	Ngamma
Obeta
Pmoving_mean
Qmoving_variance
Rregularization_losses
Strainable_variables
T	variables
U	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "BatchNormalization", "name": "bn5", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "bn5", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.9, "epsilon": 1e-06, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {"1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128]}}
?
Vregularization_losses
Wtrainable_variables
X	variables
Y	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Activation", "name": "act5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "act5", "trainable": true, "dtype": "float32", "activation": "binary_tanh"}}
?	

Zkernel
[lr_multipliers
\regularization_losses
]trainable_variables
^	variables
_	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "BinaryDense", "name": "dense6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense6", "trainable": true, "dtype": "float32", "units": 100, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "RandomUniform", "config": {"minval": -1.0, "maxval": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": {"class_name": "Clip", "config": {"min_value": -1.0, "max_value": 1.0}}, "bias_constraint": null, "H": 1.0, "kernel_lr_multiplier": 12.328827857971191, "bias_lr_multiplier": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128]}}
?	
`axis
	agamma
bbeta
cmoving_mean
dmoving_variance
eregularization_losses
ftrainable_variables
g	variables
h	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "BatchNormalization", "name": "bn6", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "bn6", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.9, "epsilon": 1e-06, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {"1": 100}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 100]}}
?

ibeta_1

jbeta_2
	kdecay
llearning_rate
miterm? m?!m?,m?7m?8m?Gm?Nm?Om?Zm?am?bm?v? v?!v?,v?7v?8v?Gv?Nv?Ov?Zv?av?bv?"
	optimizer
 "
trackable_list_wrapper
v
0
 1
!2
,3
74
85
G6
N7
O8
Z9
a10
b11"
trackable_list_wrapper
?
0
 1
!2
"3
#4
,5
76
87
98
:9
G10
N11
O12
P13
Q14
Z15
a16
b17
c18
d19"
trackable_list_wrapper
?
nnon_trainable_variables
regularization_losses
olayer_regularization_losses

players
qmetrics
rlayer_metrics
trainable_variables
	variables
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
&:$ 2conv1/kernel
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
?
snon_trainable_variables
regularization_losses
tlayer_regularization_losses

ulayers
vmetrics
wlayer_metrics
trainable_variables
	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
xnon_trainable_variables
regularization_losses
ylayer_regularization_losses

zlayers
{metrics
|layer_metrics
trainable_variables
	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
: 2	bn1/gamma
: 2bn1/beta
:  (2bn1/moving_mean
#:!  (2bn1/moving_variance
 "
trackable_list_wrapper
.
 0
!1"
trackable_list_wrapper
<
 0
!1
"2
#3"
trackable_list_wrapper
?
}non_trainable_variables
$regularization_losses
~layer_regularization_losses

layers
?metrics
?layer_metrics
%trainable_variables
&	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
(regularization_losses
 ?layer_regularization_losses
?layers
?metrics
?layer_metrics
)trainable_variables
*	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.:,  2binary_conv2d/kernel
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
,0"
trackable_list_wrapper
'
,0"
trackable_list_wrapper
?
?non_trainable_variables
.regularization_losses
 ?layer_regularization_losses
?layers
?metrics
?layer_metrics
/trainable_variables
0	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
2regularization_losses
 ?layer_regularization_losses
?layers
?metrics
?layer_metrics
3trainable_variables
4	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
':%2batch_normalization/gamma
&:$2batch_normalization/beta
/:- (2batch_normalization/moving_mean
3:1 (2#batch_normalization/moving_variance
 "
trackable_list_wrapper
.
70
81"
trackable_list_wrapper
<
70
81
92
:3"
trackable_list_wrapper
?
?non_trainable_variables
;regularization_losses
 ?layer_regularization_losses
?layers
?metrics
?layer_metrics
<trainable_variables
=	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?regularization_losses
 ?layer_regularization_losses
?layers
?metrics
?layer_metrics
@trainable_variables
A	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
Cregularization_losses
 ?layer_regularization_losses
?layers
?metrics
?layer_metrics
Dtrainable_variables
E	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
!:
?@?2dense5/kernel
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
G0"
trackable_list_wrapper
'
G0"
trackable_list_wrapper
?
?non_trainable_variables
Iregularization_losses
 ?layer_regularization_losses
?layers
?metrics
?layer_metrics
Jtrainable_variables
K	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
:?2	bn5/gamma
:?2bn5/beta
 :? (2bn5/moving_mean
$:"? (2bn5/moving_variance
 "
trackable_list_wrapper
.
N0
O1"
trackable_list_wrapper
<
N0
O1
P2
Q3"
trackable_list_wrapper
?
?non_trainable_variables
Rregularization_losses
 ?layer_regularization_losses
?layers
?metrics
?layer_metrics
Strainable_variables
T	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
Vregularization_losses
 ?layer_regularization_losses
?layers
?metrics
?layer_metrics
Wtrainable_variables
X	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 :	?d2dense6/kernel
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
Z0"
trackable_list_wrapper
'
Z0"
trackable_list_wrapper
?
?non_trainable_variables
\regularization_losses
 ?layer_regularization_losses
?layers
?metrics
?layer_metrics
]trainable_variables
^	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
:d2	bn6/gamma
:d2bn6/beta
:d (2bn6/moving_mean
#:!d (2bn6/moving_variance
 "
trackable_list_wrapper
.
a0
b1"
trackable_list_wrapper
<
a0
b1
c2
d3"
trackable_list_wrapper
?
?non_trainable_variables
eregularization_losses
 ?layer_regularization_losses
?layers
?metrics
?layer_metrics
ftrainable_variables
g	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
: (2beta_1
: (2beta_2
: (2decay
: (2learning_rate
:	 (2	Adam/iter
X
"0
#1
92
:3
P4
Q5
c6
d7"
trackable_list_wrapper
 "
trackable_list_wrapper
?
0
1
2
3
4
5
6
7
	8

9
10
11
12
13"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
"0
#1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
90
:1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
P0
Q1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
c0
d1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?

?total

?count
?	variables
?	keras_api"?
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
?

?total

?count
?
_fn_kwargs
?	variables
?	keras_api"?
_tf_keras_metric?{"class_name": "MeanMetricWrapper", "name": "acc", "dtype": "float32", "config": {"name": "acc", "dtype": "float32", "fn": "categorical_accuracy"}}
:  (2total
:  (2count
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
+:) 2Adam/conv1/kernel/m
: 2Adam/bn1/gamma/m
: 2Adam/bn1/beta/m
3:1  2Adam/binary_conv2d/kernel/m
,:*2 Adam/batch_normalization/gamma/m
+:)2Adam/batch_normalization/beta/m
&:$
?@?2Adam/dense5/kernel/m
:?2Adam/bn5/gamma/m
:?2Adam/bn5/beta/m
%:#	?d2Adam/dense6/kernel/m
:d2Adam/bn6/gamma/m
:d2Adam/bn6/beta/m
+:) 2Adam/conv1/kernel/v
: 2Adam/bn1/gamma/v
: 2Adam/bn1/beta/v
3:1  2Adam/binary_conv2d/kernel/v
,:*2 Adam/batch_normalization/gamma/v
+:)2Adam/batch_normalization/beta/v
&:$
?@?2Adam/dense5/kernel/v
:?2Adam/bn5/gamma/v
:?2Adam/bn5/beta/v
%:#	?d2Adam/dense6/kernel/v
:d2Adam/bn6/gamma/v
:d2Adam/bn6/beta/v
?2?
 __inference__wrapped_model_25509?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *2?/
-?*
conv1_input?????????@@
?2?
E__inference_sequential_layer_call_and_return_conditional_losses_26498
E__inference_sequential_layer_call_and_return_conditional_losses_27062
E__inference_sequential_layer_call_and_return_conditional_losses_27270
E__inference_sequential_layer_call_and_return_conditional_losses_26556?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
*__inference_sequential_layer_call_fn_27315
*__inference_sequential_layer_call_fn_27360
*__inference_sequential_layer_call_fn_26660
*__inference_sequential_layer_call_fn_26763?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
@__inference_conv1_layer_call_and_return_conditional_losses_25537?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *7?4
2?/+???????????????????????????
?2?
%__inference_conv1_layer_call_fn_25545?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *7?4
2?/+???????????????????????????
?2?
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_25551?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
-__inference_max_pooling2d_layer_call_fn_25557?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
>__inference_bn1_layer_call_and_return_conditional_losses_27380
>__inference_bn1_layer_call_and_return_conditional_losses_27444
>__inference_bn1_layer_call_and_return_conditional_losses_27398
>__inference_bn1_layer_call_and_return_conditional_losses_27462?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
#__inference_bn1_layer_call_fn_27411
#__inference_bn1_layer_call_fn_27424
#__inference_bn1_layer_call_fn_27475
#__inference_bn1_layer_call_fn_27488?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
?__inference_act1_layer_call_and_return_conditional_losses_27508?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
$__inference_act1_layer_call_fn_27513?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
H__inference_binary_conv2d_layer_call_and_return_conditional_losses_25689?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *7?4
2?/+??????????????????????????? 
?2?
-__inference_binary_conv2d_layer_call_fn_25697?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *7?4
2?/+??????????????????????????? 
?2?
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_25703?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
/__inference_max_pooling2d_1_layer_call_fn_25709?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
N__inference_batch_normalization_layer_call_and_return_conditional_losses_27615
N__inference_batch_normalization_layer_call_and_return_conditional_losses_27533
N__inference_batch_normalization_layer_call_and_return_conditional_losses_27597
N__inference_batch_normalization_layer_call_and_return_conditional_losses_27551?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
3__inference_batch_normalization_layer_call_fn_27641
3__inference_batch_normalization_layer_call_fn_27577
3__inference_batch_normalization_layer_call_fn_27628
3__inference_batch_normalization_layer_call_fn_27564?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
E__inference_activation_layer_call_and_return_conditional_losses_27661?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_activation_layer_call_fn_27666?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
B__inference_flatten_layer_call_and_return_conditional_losses_27672?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
'__inference_flatten_layer_call_fn_27677?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
A__inference_dense5_layer_call_and_return_conditional_losses_27704?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
&__inference_dense5_layer_call_fn_27711?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
>__inference_bn5_layer_call_and_return_conditional_losses_27767
>__inference_bn5_layer_call_and_return_conditional_losses_27747?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
#__inference_bn5_layer_call_fn_27780
#__inference_bn5_layer_call_fn_27793?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
?__inference_act5_layer_call_and_return_conditional_losses_27813?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
$__inference_act5_layer_call_fn_27818?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
A__inference_dense6_layer_call_and_return_conditional_losses_27845?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
&__inference_dense6_layer_call_fn_27852?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
>__inference_bn6_layer_call_and_return_conditional_losses_27888
>__inference_bn6_layer_call_and_return_conditional_losses_27908?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
#__inference_bn6_layer_call_fn_27921
#__inference_bn6_layer_call_fn_27934?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
#__inference_signature_wrapper_26818conv1_input"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 ?
 __inference__wrapped_model_25509 !"#,789:GQNPOZdacb<?9
2?/
-?*
conv1_input?????????@@
? ")?&
$
bn6?
bn6?????????d?
?__inference_act1_layer_call_and_return_conditional_losses_27508h7?4
-?*
(?%
inputs?????????   
? "-?*
#? 
0?????????   
? ?
$__inference_act1_layer_call_fn_27513[7?4
-?*
(?%
inputs?????????   
? " ??????????   ?
?__inference_act5_layer_call_and_return_conditional_losses_27813Z0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? u
$__inference_act5_layer_call_fn_27818M0?-
&?#
!?
inputs??????????
? "????????????
E__inference_activation_layer_call_and_return_conditional_losses_27661h7?4
-?*
(?%
inputs????????? 
? "-?*
#? 
0????????? 
? ?
*__inference_activation_layer_call_fn_27666[7?4
-?*
(?%
inputs????????? 
? " ?????????? ?
N__inference_batch_normalization_layer_call_and_return_conditional_losses_27533r789:;?8
1?.
(?%
inputs????????? 
p
? "-?*
#? 
0????????? 
? ?
N__inference_batch_normalization_layer_call_and_return_conditional_losses_27551r789:;?8
1?.
(?%
inputs????????? 
p 
? "-?*
#? 
0????????? 
? ?
N__inference_batch_normalization_layer_call_and_return_conditional_losses_27597?789:M?J
C?@
:?7
inputs+???????????????????????????
p
? "??<
5?2
0+???????????????????????????
? ?
N__inference_batch_normalization_layer_call_and_return_conditional_losses_27615?789:M?J
C?@
:?7
inputs+???????????????????????????
p 
? "??<
5?2
0+???????????????????????????
? ?
3__inference_batch_normalization_layer_call_fn_27564e789:;?8
1?.
(?%
inputs????????? 
p
? " ?????????? ?
3__inference_batch_normalization_layer_call_fn_27577e789:;?8
1?.
(?%
inputs????????? 
p 
? " ?????????? ?
3__inference_batch_normalization_layer_call_fn_27628?789:M?J
C?@
:?7
inputs+???????????????????????????
p
? "2?/+????????????????????????????
3__inference_batch_normalization_layer_call_fn_27641?789:M?J
C?@
:?7
inputs+???????????????????????????
p 
? "2?/+????????????????????????????
H__inference_binary_conv2d_layer_call_and_return_conditional_losses_25689?,I?F
??<
:?7
inputs+??????????????????????????? 
? "??<
5?2
0+??????????????????????????? 
? ?
-__inference_binary_conv2d_layer_call_fn_25697?,I?F
??<
:?7
inputs+??????????????????????????? 
? "2?/+??????????????????????????? ?
>__inference_bn1_layer_call_and_return_conditional_losses_27380? !"#M?J
C?@
:?7
inputs+????????? ??????????????????
p
? "??<
5?2
0+????????? ??????????????????
? ?
>__inference_bn1_layer_call_and_return_conditional_losses_27398? !"#M?J
C?@
:?7
inputs+????????? ??????????????????
p 
? "??<
5?2
0+????????? ??????????????????
? ?
>__inference_bn1_layer_call_and_return_conditional_losses_27444r !"#;?8
1?.
(?%
inputs?????????   
p
? "-?*
#? 
0?????????   
? ?
>__inference_bn1_layer_call_and_return_conditional_losses_27462r !"#;?8
1?.
(?%
inputs?????????   
p 
? "-?*
#? 
0?????????   
? ?
#__inference_bn1_layer_call_fn_27411? !"#M?J
C?@
:?7
inputs+????????? ??????????????????
p
? "2?/+????????? ???????????????????
#__inference_bn1_layer_call_fn_27424? !"#M?J
C?@
:?7
inputs+????????? ??????????????????
p 
? "2?/+????????? ???????????????????
#__inference_bn1_layer_call_fn_27475e !"#;?8
1?.
(?%
inputs?????????   
p
? " ??????????   ?
#__inference_bn1_layer_call_fn_27488e !"#;?8
1?.
(?%
inputs?????????   
p 
? " ??????????   ?
>__inference_bn5_layer_call_and_return_conditional_losses_27747dPQNO4?1
*?'
!?
inputs??????????
p
? "&?#
?
0??????????
? ?
>__inference_bn5_layer_call_and_return_conditional_losses_27767dQNPO4?1
*?'
!?
inputs??????????
p 
? "&?#
?
0??????????
? ~
#__inference_bn5_layer_call_fn_27780WPQNO4?1
*?'
!?
inputs??????????
p
? "???????????~
#__inference_bn5_layer_call_fn_27793WQNPO4?1
*?'
!?
inputs??????????
p 
? "????????????
>__inference_bn6_layer_call_and_return_conditional_losses_27888bcdab3?0
)?&
 ?
inputs?????????d
p
? "%?"
?
0?????????d
? ?
>__inference_bn6_layer_call_and_return_conditional_losses_27908bdacb3?0
)?&
 ?
inputs?????????d
p 
? "%?"
?
0?????????d
? |
#__inference_bn6_layer_call_fn_27921Ucdab3?0
)?&
 ?
inputs?????????d
p
? "??????????d|
#__inference_bn6_layer_call_fn_27934Udacb3?0
)?&
 ?
inputs?????????d
p 
? "??????????d?
@__inference_conv1_layer_call_and_return_conditional_losses_25537?I?F
??<
:?7
inputs+???????????????????????????
? "??<
5?2
0+??????????????????????????? 
? ?
%__inference_conv1_layer_call_fn_25545?I?F
??<
:?7
inputs+???????????????????????????
? "2?/+??????????????????????????? ?
A__inference_dense5_layer_call_and_return_conditional_losses_27704]G0?-
&?#
!?
inputs??????????@
? "&?#
?
0??????????
? z
&__inference_dense5_layer_call_fn_27711PG0?-
&?#
!?
inputs??????????@
? "????????????
A__inference_dense6_layer_call_and_return_conditional_losses_27845\Z0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????d
? y
&__inference_dense6_layer_call_fn_27852OZ0?-
&?#
!?
inputs??????????
? "??????????d?
B__inference_flatten_layer_call_and_return_conditional_losses_27672a7?4
-?*
(?%
inputs????????? 
? "&?#
?
0??????????@
? 
'__inference_flatten_layer_call_fn_27677T7?4
-?*
(?%
inputs????????? 
? "???????????@?
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_25703?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
/__inference_max_pooling2d_1_layer_call_fn_25709?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_25551?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
-__inference_max_pooling2d_layer_call_fn_25557?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
E__inference_sequential_layer_call_and_return_conditional_losses_26498? !"#,789:GPQNOZcdabD?A
:?7
-?*
conv1_input?????????@@
p

 
? "%?"
?
0?????????d
? ?
E__inference_sequential_layer_call_and_return_conditional_losses_26556? !"#,789:GQNPOZdacbD?A
:?7
-?*
conv1_input?????????@@
p 

 
? "%?"
?
0?????????d
? ?
E__inference_sequential_layer_call_and_return_conditional_losses_27062~ !"#,789:GPQNOZcdab??<
5?2
(?%
inputs?????????@@
p

 
? "%?"
?
0?????????d
? ?
E__inference_sequential_layer_call_and_return_conditional_losses_27270~ !"#,789:GQNPOZdacb??<
5?2
(?%
inputs?????????@@
p 

 
? "%?"
?
0?????????d
? ?
*__inference_sequential_layer_call_fn_26660v !"#,789:GPQNOZcdabD?A
:?7
-?*
conv1_input?????????@@
p

 
? "??????????d?
*__inference_sequential_layer_call_fn_26763v !"#,789:GQNPOZdacbD?A
:?7
-?*
conv1_input?????????@@
p 

 
? "??????????d?
*__inference_sequential_layer_call_fn_27315q !"#,789:GPQNOZcdab??<
5?2
(?%
inputs?????????@@
p

 
? "??????????d?
*__inference_sequential_layer_call_fn_27360q !"#,789:GQNPOZdacb??<
5?2
(?%
inputs?????????@@
p 

 
? "??????????d?
#__inference_signature_wrapper_26818? !"#,789:GQNPOZdacbK?H
? 
A?>
<
conv1_input-?*
conv1_input?????????@@")?&
$
bn6?
bn6?????????d