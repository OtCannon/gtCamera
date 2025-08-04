
#include "PropControlBase.h"

#include <QMessageBox>
#include <QCheckBox>

namespace ic4::ui
{
	using BooleanCheckBox = app::CaptureFocus<QCheckBox>;

	class PropBooleanControl : public PropControlBase<ic4::PropBoolean>
	{
		BooleanCheckBox* check_;

	public:
		PropBooleanControl(ic4::PropBoolean prop, QWidget* parent, ic4::Grabber* grabber)
			: PropControlBase(prop, parent, grabber)
		{
			check_ = new BooleanCheckBox(this);
			check_->setText("");
			check_->focus_in += [this](auto*) { onPropSelected(); };

			// use stylesheet in qss - breaks checkbox images used in qss!
			//check_->setStyleSheet("QCheckBox::indicator { width: 16px; height: 16px; }");

			connect(check_, &QCheckBox::stateChanged, this, &PropBooleanControl::check);

			update_all();

			layout_->addWidget(check_);
			layout_->setContentsMargins(8, 8, 0, 8);
		}

	private:
		void check(int new_state)
		{
			ic4::Error err;
			if (!propSetValue(new_state == Qt::Checked, err, &PropBoolean::setValue))
			{
				QMessageBox::critical(this, {}, err.message().c_str());
			}
		}

		void update_all() override
		{
			check_->setEnabled(!shoudDisplayAsLocked() && !prop_.isReadOnly(ic4::Error::Ignore()));
			check_->blockSignals(true);

			ic4::Error err;
			auto value = prop_.getValue(err);
			if( err.isSuccess() )
			{
				check_->setChecked(value);
			}
			else
			{
				qWarning() << "Error " << prop_.name(ic4::Error::Ignore()).c_str() << " in update_all " << err.message().c_str();
			}

			check_->blockSignals(false);
		}
	};
}
