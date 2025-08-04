
#include "PropControlBase.h"
#include "../Event.h"

#include <QKeyEvent>
#include <QLineEdit>
#include <QMessageBox>

namespace ic4::ui
{
	class PropStringControl : public PropControlBase<ic4::PropString>
	{
		class StringLineEdit : public app::CaptureFocus<QLineEdit>
		{
		public:
			StringLineEdit(QWidget* parent)
				: app::CaptureFocus<QLineEdit>(parent)
			{
			}

			app::Event<> escapePressed;
		protected:
			void keyPressEvent(QKeyEvent* e) override
			{
				if (e->key() == Qt::Key_Enter)
				{
					editingFinished();
					return;
				}
				if (e->key() == Qt::Key_Escape)
				{
					escapePressed(nullptr);
					return;
				}

				QLineEdit::keyPressEvent(e);
			}
		};

	private:
		StringLineEdit* edit_;

	public:
		PropStringControl(ic4::PropString prop, QWidget* parent, ic4::Grabber* grabber)
			: PropControlBase(prop, parent, grabber)
		{
			uint64_t max_length = (uint64_t)-1;
			try
			{
				max_length = prop.maxLength();
			}
			catch (const ic4::IC4Exception& iex)
			{
				qDebug() << "Error " << prop.name().c_str() << " in " << iex.what();
			}


			edit_ = new StringLineEdit(this);
			edit_->setReadOnly(prop.isReadOnly());
			connect(edit_, &QLineEdit::editingFinished, this, &PropStringControl::set_value);
			edit_->escapePressed += [this](auto) { update_value(); };
			edit_->setMaxLength(max_length);
			edit_->focus_in += [this](auto*) { onPropSelected(); };

			update_all();

			layout_->addWidget(edit_);
		}

	private:
		void set_value()
		{
			if (edit_->isReadOnly())
				return;

			auto new_val = edit_->text().toStdString();

			ic4::Error err;
			if (!propSetValue(new_val, err, &PropString::setValue))
			{
				QMessageBox::critical(this, {}, err.message().c_str());
			}
		}

		void update_value()
		{
			edit_->blockSignals(true);
			try
			{
				auto val = prop_.getValue();
				edit_->setText(QString::fromStdString(val));
			}
			catch (const ic4::IC4Exception& iex)
			{
				qDebug() << "Error " << prop_.name(ic4::Error::Ignore()).c_str() << " in update_value() " << iex.what();
				edit_->setText("<Error>");
			}

			edit_->blockSignals(false);
		}

	protected:
		void update_all() override
		{
			update_value();

			edit_->blockSignals(true);

			bool is_readonly = prop_.isReadOnly(ic4::Error::Ignore());
			bool is_locked = shoudDisplayAsLocked();

			edit_->setSelection(0, 0);
			edit_->setReadOnly(is_readonly || is_locked);

			// use StyleSheet in qss!
			//if (is_readonly || is_locked)
			//{
			//	edit_->setStyleSheet(R"(background-color: palette(window);)");
			//}
			//else
			//{
			//	edit_->setStyleSheet(R"(background-color: palette(base);)");
			//}
			edit_->blockSignals(false);
			edit_->update();
		}
	};
}