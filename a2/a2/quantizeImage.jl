using PyPlot
include("kMeans.jl")
dog = imread("dog.png")

function quantizeImage(img, b)
  (nRows,nCols,m) = size(img)
  assert(m.==3)
  z = nRows * nCols

  P = reshape(img,z,3)
  model = kMeans(P,2.^b;doPlot=false)
  y = model.predict(P)

  return deQuantizeImage(nRows,nCols,y,model.W)
end


function deQuantizeImage(nRows,nCols,y,W)
  y = reshape(y,nRows,nCols)
  n = zeros(nRows,nCols,3)

  for i in 1:nRows
    for j in 1:nCols
      n[i,j,:] = W[y[i,j],:]
    end
  end

  return n
end

  imshow(quantizeImage(dog,6))
