set(MI_PLUGIN_PREFIX "volumes")

add_plugin(constvolume  const.cpp)
add_plugin(gridvolume   grid.cpp)
add_plugin(toroidalvolume toroidal.cpp)

set(MI_PLUGIN_TARGETS "${MI_PLUGIN_TARGETS}" PARENT_SCOPE)
