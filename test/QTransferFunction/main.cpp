#include <QtWidgets/qapplication.h>

#include <CMakeIn.h>
#include <util/VolumeCfg.h>
#include <util/QTransferFunction.h>

using namespace kouek;

int main(int argc, char** argv)
{
	QApplication app(argc, argv);

	VolumeConfig cfg(std::string(PROJECT_SOURCE_DIR)
		+ "/cfg/VolumeCfg.json");
	QTransferFunctionWidget view;
	std::map<uint8_t, std::array<qreal, 4>> tfDat;
	for (auto& tfPt : cfg.getTF().points)
		tfDat.emplace(std::piecewise_construct,
			std::forward_as_tuple(tfPt.key),
			std::forward_as_tuple(tfPt.value));
	view.loadFromTFData(tfDat);
	view.show();

	return app.exec();
}
