# Iris flowers

## The dataset
The Iris dataset was published by the statistician Ronald Fisher in 1936, using measurements collected by the botanist Edgar Anderson. It has 150 flowers, 50 from each of three species: Iris setosa, Iris versicolor and Iris virginica. Each flower has four measurements in centimetres: sepal length, sepal width, petal length and petal width.

## Telling the species apart
Petal measurements are the most useful features. Setosa has very small petals (length about 1.0 to 1.9 cm, width about 0.1 to 0.6 cm) and is linearly separable from the other two species, so even a simple model classifies it perfectly. Versicolor has medium petals (length about 3.0 to 5.1 cm, width about 1.0 to 1.8 cm). Virginica has the largest petals (length about 4.5 to 6.9 cm, width about 1.4 to 2.5 cm).

## Where models make mistakes
Versicolor and virginica overlap in feature space. A flower with petal length around 4.8 to 5.1 cm and petal width around 1.6 to 1.8 cm sits on the boundary, and almost all classification errors on Iris involve these two species. Setosa is practically never confused with the others.

## Sepals
Sepals are the outer green leaf-like parts of the flower bud, petals are the coloured inner parts. Setosa tends to have wider but shorter sepals than the other two species. Sepal measurements alone separate the species less well than petal measurements.
