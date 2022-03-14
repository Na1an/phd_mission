# How to config/use CGAL - demo Polyhedron

## 1. Download the newest version CGAL

(see https://doc.cgal.org/latest/Manual/index.html)

* Go to release page : https://github.com/CGAL/cgal/releases
* Chose & Download `CGAL-X.X.tar.xz` package 

* And then tape ` tar xf CGAL-5.4.tar.xz`

## 2. Config

```
cd $HOME/CGAL-5.4
mkdir build
cd build
cmake ..                                                                          # configure CGAL
make install                                                                      # install CGAL
cd examples/Triangulation_2                                                       # go to an example directory
cmake -DCGAL_DIR=$CMAKE_INSTALLED_PREFIX/lib/CGAL -DCMAKE_BUILD_TYPE=Release .    # configure the examples
make                                                                              # build the examples
```



## 3. Open it!

* `./Polyhedron_3`



## 4. Debug

You will find many error when you try to compile the demo : **Polyhedron**. What I have encountered is as follows : 



## 5. Data Format

It seems like CGAL doesn't support .las/.laz format even we download **lastool**. So I suggest **.las -> .ply **.