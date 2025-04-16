# bagofholding

`bagofholding` is designed to be an easy stand-in for `pickle` serialization for python object that is transparent, flexible, and suitable for long-term storage.

## Advantages
### Drop-in replacement

`bagofholding` stores (almost) any `pickle`-able python object, and can be easily used as a drop-in replacement for `pickle` serialization:

```python
>>> import bagofholding as boh
>>>
>>> boh.H5Bag.save(42, "file.h5")
>>> print(boh.H5Bag("file.h5").load())
42

```


### Browseable

The contents of stored objects can be browsed without actually re-instantiating any of the stored data.
In the example above, we saw that saving is a class-method, while loading is an instance method.
We can grab the "bag" instance and use it to peek at what's inside!

Let's use a slightly more complex object.
Readers familiar with `pickle` will be able to see that the "reduced" structure of the object is captured in the structure of the storage itself:

```python
>>> class MyThing:
...     def __init__(self, answer: int, question: str):
...         self.answer = answer
...         self.question = question
>>>
>>> something = MyThing(42, "still computing...")
>>> boh.H5Bag.save(something, "something.h5")
>>> bag = boh.H5Bag("something.h5")
>>> bag.list_paths()
['object', 'object/args', 'object/args/i0', 'object/constructor', 'object/item_iterator', 'object/kv_iterator', 'object/state', 'object/state/answer', 'object/state/question']

```

Item-access on the bag object gives access to metadata stored alongside the actual serialized information:

```python
>>> bag["object"]
Metadata(content_type='bagofholding.content.Reducible', qualname='MyThing', module='__main__', version=None, meta=None)

```

For Jupyter users, we power-up browsing capabilities with a widget under `bag.browse()` which lets you navigate the tree and see both metadata values and stored types:

![](_static/widget_snapshot.png)


### Partial-loading

Stored objects can also be re-instantiated _in part_ by leveraging their storage path:

```python
>>> bag.load("object/state/answer")
42

```

Note that we didn't re-instantiate any part of the object other than this one integer!

This feature is incredibly useful for long-term storage and data transferability, as the loading environment does not need to fully match the saving environment -- only the environment required to load the actual piece of data desired matches.
Consider some complex object which, ultimately, contains important or expensive-to-calculate numeric data, e.g. in the form of numpy array.
With `bagofholding`, you can pass this data to a colleague running a different python environment, or come back to it years later.
With only `bagofholding` and `numpy` installed, the end user can browse through the stored object, access, and load only the valuable numeric data without re-installing the entire original environment.


### Version control

In the examples above, we saw that version (and of course package) information is part of the stored metadata.
This is useful post-facto for knowing what packages need to be installed to properly load your serialized data.
You can also specify at load-time how strict or relaxed `bagofholding` should be in re-instantiating data if a stored version does not match the currently installed version, thus protecting you from flawed re-instantiations.

`bagofholding` also provides tools to act on this data a-priori.
To increase the likelihood that stored data will be accessible in the future, you can outlaw any (sub)objects coming from particular modules:

```python
import bagofholding.exception
>> > try:
    ...
boh.H5Bag.save(something, "will_fail.h5", forbidden_modules=("__main__",))
... except bagofholding.exception.ModuleForbiddenError as e:
...
print(e)
Module
'__main__' is forbidden as a
source
of
stored
objects.Change
the
`forbidden_modules` or move
this
object
to
an
allowed
module.

```

And/or demand that all objects have an identifiable version that:

```python
import bagofholding.exception
>> > try:
    ...
boh.H5Bag.save(something, "will_fail.h5", require_versions=True)
... except bagofholding.exception.NoVersionError as e:
...
print(e)
Could
not find
a
version
for __main__.Either disable `require_versions`, use `version_scraping` to find an existing version for this package, or add versioning to the unversioned package.

```

Of course, metadata for the bag itself is also stored.
We saw this in the GUI snapshot above, but it can also be accessed directly by code:

```python
>>> boh.H5Bag.get_bag_info()
H5Info(qualname='H5Bag', module='bagofholding.h5.bag', version='...', libver_str='latest')

```

(In reality you will see a version code, it is omitted here because this example is executed automatically in the test suite.)

## Going further

For a more in-depth look at the above features and to explore other aspects of `bagofholding`, check out [the tutorial notebook](../notebooks/tutorial.ipynb).

Finally, `bagofholding` prioritizes transparency in what is stored and ease-of-use for both savers and loaders/browsers.
As such, the current hdf5-based implementation is likely to be significantly less performant than raw pickling, due to the creation of many small datasets that allow the h5 file to directly replicate the underlying structure of the python objects being saved.
For objects which contain large `numpy` arrays, this disadvantage is significantly alleviated as we benefit from the very efficient treatment of such arrays in hdf5 and `h5py`.
For all other objects, the current `bagofholding.H5Bag` is still an appropriate choice when the robustness of long term storage is more pressing than optimizing storage space.
Other bag types may be available in the future.
