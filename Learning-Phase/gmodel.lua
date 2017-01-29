require 'nn'
require 'cunn'
gpu=1
noutputs=4
nfeats=8

nstates={16,32,64,256}
filter={9,5}
stride={3,1}
poolsize=2

function create_network()

	modelA=nn.Sequential()
	--cov1
	modelA:add(nn.SpatialConvolution(nfeats, nstates[1],filter[1],filter[1],stride[1],stride[1],1))
	modelA:add(nn.ReLU())
	modelA:add(nn.SpatialMaxPooling(poolsize,poolsize,poolsize,poolsize))
	--cov2
	modelA:add(nn.SpatialConvolution(nstates[1],nstates[2],filter[2],filter[2],stride[2],stride[2]))
	modelA:add(nn.ReLU())
	modelA:add(nn.SpatialMaxPooling(poolsize,poolsize,poolsize,poolsize))
	--cov3
	modelA:add(nn.SpatialConvolution(nstates[2],nstates[3],filter[2],filter[2],stride[2],stride[2]))
	modelA:add(nn.ReLU())
	modelA:add(nn.SpatialMaxPooling(poolsize,poolsize,poolsize,poolsize))
	--RESHAPE
	modelA:add(nn.View(nstates[3]*filter[2]*filter[2]))
	modelA:add(nn.Linear(nstates[3]*filter[2]*filter[2],nstates[4]))
	modelA:add(nn.ReLU())
	modelA:add(nn.Linear(nstates[4],noutputs))
	modelA=modelA:cuda()


	modelB=nn.Sequential()
	--cov1
	modelB:add(nn.SpatialConvolution(nfeats, nstates[1],filter[1],filter[1],stride[1],stride[1],1))
	modelB:add(nn.ReLU())
	modelB:add(nn.SpatialMaxPooling(poolsize,poolsize,poolsize,poolsize))
	--cov2
	modelB:add(nn.SpatialConvolution(nstates[1],nstates[2],filter[2],filter[2],stride[2],stride[2]))
	modelB:add(nn.ReLU())
	modelB:add(nn.SpatialMaxPooling(poolsize,poolsize,poolsize,poolsize))
	--cov3
	modelB:add(nn.SpatialConvolution(nstates[2],nstates[3],filter[2],filter[2],stride[2],stride[2]))
	modelB:add(nn.ReLU())
	modelB:add(nn.SpatialMaxPooling(poolsize,poolsize,poolsize,poolsize))
	--RESHAPE
	modelB:add(nn.View(nstates[3]*filter[2]*filter[2]))
	modelB:add(nn.Linear(nstates[3]*filter[2]*filter[2],nstates[4]))
	modelB:add(nn.ReLU())

	modelB:add(nn.Linear(nstates[4],noutputs))

	modelB=modelB:cuda()

return modelA,modelB
end
