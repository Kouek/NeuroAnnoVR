#ifndef KOUEK_VOLUME_CFG_H
#define KOUEK_VOLUME_CFG_H

#include <fstream>
#include <sstream>

#include <json.hpp>

namespace kouek
{
	class VolumeConfig
	{
	private:
		float spaceX, spaceY, spaceZ;
		float baseSpace;
		std::string resourcePath;
		nlohmann::json json;

	public:
		/// <summary>
		/// Init Volume Config from volumeCfgPath.
		/// </summary>
		/// <param name="volumeCfgPath">absolute Path of Volume Config File</param>
		VolumeConfig(const char* volumeCfgPath)
		{
			using namespace std;
			using namespace nlohmann;

			ifstream in(volumeCfgPath);
			if (!in.is_open()) throw runtime_error("Cannot open file: " + string(volumeCfgPath));

			try
			{
				in >> json;
				in.close();

				auto spaces = json.at("space");
				spaceX = spaces[0];
				spaceY = spaces[1];
				spaceZ = spaces[2];
				baseSpace = min({ spaceX, spaceY, spaceZ });

				auto screenJson = json.at("screen").at("0");
				resourcePath = screenJson.at("resourcePath");
			}
			catch (json::parse_error& e)
			{
				throw runtime_error("Problem occurs at byte: " + std::to_string(e.byte) + ". Problem is: " + e.what());
			}
		}
#define GETTER(retType, firstChInLowerCase, firstChInUpperCase, successor)                                             \
    retType get##firstChInUpperCase##successor() const                                                                 \
    {                                                                                                                  \
        return firstChInLowerCase##successor;                                                                          \
    }
		GETTER(float, s, S, paceX)
			GETTER(float, s, S, paceY)
			GETTER(float, s, S, paceZ)
			GETTER(float, b, B, aseSpace)
			GETTER(const std::string&, r, R, esourcePath)
#undef GETTER
	};

}

#endif // !KOUEK_VOLUME_CFG_H
