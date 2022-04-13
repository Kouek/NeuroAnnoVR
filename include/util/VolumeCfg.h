#ifndef KOUEK_VOLUME_CFG_H
#define KOUEK_VOLUME_CFG_H

#include <string_view>

#include <fstream>
#include <sstream>

#include <json.hpp>

#include <VolumeSlicer/transfer_function.hpp>

namespace kouek
{
	class VolumeConfig
	{
	private:
		float spaceX, spaceY, spaceZ;
		float baseSpace;
		std::string resourcePath;
		nlohmann::json json;

		vs::TransferFunc tf;

	public:
		/// <summary>
		/// Init Volume Config from volumeCfgPath.
		/// </summary>
		/// <param name="volumeCfgPath">absolute Path of Volume Config File</param>
		VolumeConfig(std::string_view volumeCfgPath)
		{
			using namespace std;
			using namespace nlohmann;

			ifstream in(volumeCfgPath.data());
			if (!in.is_open()) throw runtime_error("Cannot open file: " + string(volumeCfgPath));

			try
			{
				in >> json;
				in.close();

				auto jsonItem = json.at("space");
				spaceX = jsonItem[0];
				spaceY = jsonItem[1];
				spaceZ = jsonItem[2];
				baseSpace = min({ spaceX, spaceY, spaceZ });

				jsonItem = json.at("screen").at("0");
				resourcePath = jsonItem.at("resourcePath");

				jsonItem = json.at("tf");
				for (auto& [tfPnt, tfCol] : jsonItem.items())
				{
					tf.points.emplace_back((uint8_t)stoi(tfPnt),
						std::array<double, 4>{tfCol[0], tfCol[1], tfCol[2], tfCol[3]});
				}
			}
			catch (json::parse_error& e)
			{
				throw runtime_error("Problem occurs at byte: "
					+ std::to_string(e.byte)
					+ ". Problem is: " + e.what());
			}
		}
		const vs::TransferFunc& getTF() const
		{
			return tf;
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
