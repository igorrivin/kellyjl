using LinearAlgebra
using Random

function ratios(v)
	return v[:, 1]./v[:, 2]
end

function dojac(features)
	featlen = size(features)[1]
	return (cat(features, ones(featlen), dims = 2))
end

function dopregrad(preds, vals)
	rats = ratios(vals)
	rats2 = rats .* rats
	term1 = rats
	term2 = rats2 .* preds
	thedot = dot(rats2, preds)
	term3 = thedot * rats
	return term1 - term2 + term3
end

function dograd(preds, vals, features)
	pregrad = dopregrad(preds, vals)
	thejac = dojac(features)
	return thejac * pregrad
end

function newton_step(params, grad, hess)
	return params - hess \ grad
end

function make_hess(preds, vals, features)
	ratios = vals[:, 1] ./ vals[:, 2]
	thejac = dojac(features)
	newjac = ratios .* thejac
	featlen = size(features)[1]
	theones = ones(featlen)
	term1 = newjac' * newjac
	prod2 = theones' * newjac
	term2 = prod2' * prod2
	return -term1 + term2/featlen
end
