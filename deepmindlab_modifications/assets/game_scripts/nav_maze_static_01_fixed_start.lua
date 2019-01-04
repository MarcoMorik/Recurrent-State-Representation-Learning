local factory = require 'factories.seek_avoid_factory'

return factory.createLevelApi{
    mapName = 'nav_maze_static_01_fixed_start',
    episodeLengthSeconds = 1000
}
