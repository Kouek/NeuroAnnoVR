#ifndef KOUEK_EDITOR_WINDOW_H
#define KOUEK_EDITOR_WINDOW_H

#include "VRView.h"

namespace Ui
{
	class EditorWindow;
}

namespace kouek
{
	class EditorWindow : public QWidget
	{
		Q_OBJECT

	private:
		VRView* vrView;
		Ui::EditorWindow* ui;

	public:
		explicit EditorWindow(QWidget* parent = Q_NULLPTR);
		~EditorWindow();

		inline VRView* getVRView()
		{
			return vrView;
		}

	signals:
		void closed();
		void reloadTFBtnClicked();

	protected:
		virtual void closeEvent(QCloseEvent* e) override;
	};
}

#endif // !KOUEK_EDITOR_WINDOW_H
