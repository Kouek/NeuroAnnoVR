#include <util/SWCConverter.h>

using namespace kouek;

static GLuint rootID = 0;
// 0
// +-1-2
// +-3-4-5
// | +-6-7
// +-8

// 9
// +-10-11
// +-12-13-14
// | +-15-16
// +-17
static std::vector<std::vector<std::vector<GLuint>>> paths = { {
		{0,1,2},{1,3,4,5},{3,6,7},{0,8}
	},{
		{9,10,11},{10,12,13,14},{12,15,16},{9,17}
	}
};

static void testWrite()
{
	FileSWC swc("./SWC.txt");
	GLPathRenderer pathRenderer;
	for (const auto& path : paths)
	{
		auto id = pathRenderer.addPath(glm::vec3{ 1.f }, glm::vec3{ 0 });
		pathRenderer.startPath(id);
		for (const auto& subPath : path)
		{
			auto id = pathRenderer.addSubPath();
			for (const auto& vert : subPath)
			{
			}
		}
	}
}

static void testRead()
{
	FileSWC swc("./SWC.txt");
	// TODO
}

int main()
{
	testWrite();
	testRead();
	return 0;
}
