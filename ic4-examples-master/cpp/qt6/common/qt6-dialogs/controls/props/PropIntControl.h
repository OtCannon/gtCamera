
#pragma once

#include "../Event.h"

#include "PropIntSpinBox.h"
#include "PropIntSlider.h"
#include "PropControlBase.h"

#include <ic4/ic4.h>

#include <QWidget>
#include <QCheckBox>
#include <QSlider>
#include <QMessageBox>
#include <QAbstractSlider>
#include <QSignalBlocker>

#include <algorithm>

namespace ic4::ui
{
	using IntCheckBox = app::CaptureFocus<QCheckBox>;
	using IntLineEdit = app::CaptureFocus<QLineEdit>;

	class PropIntControl : public PropControlBase<ic4::PropInteger>
	{
	private:
		IntCheckBox* check_ = nullptr;
		PropIntSlider* slider_ = nullptr;
		PropIntSpinBox* spin_ = nullptr;
		IntLineEdit* edit_ = nullptr;

		ic4::PropIntRepresentation representation_;
		int64_t min_;
		int64_t max_;
		ic4::PropIncrementMode inc_mode_;
		int64_t inc_;
		std::vector<int64_t> valid_value_set_;
		int64_t val_;

	private:
		static QString format_ip(uint32_t ip)
		{
			return QString("%1.%2.%3.%4")
				.arg((ip >> 24) & 0xFF)
				.arg((ip >> 16) & 0xFF)
				.arg((ip >> 8) & 0xFF)
				.arg((ip >> 0) & 0xFF);
		}
		static QString format_mac(uint64_t mac)
		{
			return QString("%1:%2:%3:%4:%5:%6")
				.arg((mac >> 40) & 0xFF, 2, 16, (QChar)'0')
				.arg((mac >> 32) & 0xFF, 2, 16, (QChar)'0')
				.arg((mac >> 24) & 0xFF, 2, 16, (QChar)'0')
				.arg((mac >> 16) & 0xFF, 2, 16, (QChar)'0')
				.arg((mac >> 8) & 0xFF, 2, 16, (QChar)'0')
				.arg((mac >> 0) & 0xFF, 2, 16, (QChar)'0');
		}

	public:
		static QString value_to_string(int64_t val, ic4::PropIntRepresentation rep)
		{
			switch (rep)
			{
			case ic4::PropIntRepresentation::Boolean:
				return val ? "True" : "False";
			case ic4::PropIntRepresentation::HexNumber:
				return QString("0x%1").arg(val, 0, 16);
			case ic4::PropIntRepresentation::PureNumber:
			case ic4::PropIntRepresentation::Linear:
			case ic4::PropIntRepresentation::Logarithmic:
			default:
				return QString::number(val);			
			case ic4::PropIntRepresentation::MACAddress:
				return format_mac(val);
			case ic4::PropIntRepresentation::IPV4Address:
				return format_ip(val);
			}
		}

	private:
		void set_value_unchecked(int64_t new_val)
		{
			ic4::Error err;
			if (!propSetValue(new_val, err, &ic4::PropInteger::setValue))
			{
				QMessageBox::critical(this, {}, err.message().c_str());
			}
		}

		void value_step(int64_t step)
		{
			int64_t new_val = val_;

			if (inc_mode_ == ic4::PropIncrementMode::ValueSet)
			{
				auto it = std::lower_bound(valid_value_set_.begin(), valid_value_set_.end(), new_val);

				int64_t max_increase = std::distance(it, std::prev(valid_value_set_.end()));
				int64_t max_decrease = std::distance(it, valid_value_set_.begin());
				auto adjust = std::clamp(step, max_decrease, max_increase);

				std::advance(it, adjust);
				new_val = *it;
			}
			else
			{
				step *= inc_;

				if (step < 0)
				{
					if (val_ > min_ - step)
						new_val = val_ + step;
					else
						new_val = min_;
				}
				else if (step > 0)
				{
					if (val_ < max_ - step)
						new_val = val_ + step;
					else
						new_val = max_;
				}
			}

			set_value_unchecked(new_val);
		}

		void set_value(int new_pos)
		{
			int64_t new_val = std::clamp(static_cast<int64_t>(new_pos), min_, max_);

			if (inc_mode_ == ic4::PropIncrementMode::ValueSet)
			{
				auto it = std::upper_bound(valid_value_set_.begin(), valid_value_set_.end(), new_val);
				if (it == valid_value_set_.end())
				{
					new_val = valid_value_set_.back();
				}
				else if (it == valid_value_set_.begin())
				{
					new_val = valid_value_set_.front();
				}
				else
				{
					new_val = *std::prev(it);
				}
			}
			else
			{
				if ((new_val - min_) % inc_)
				{
					auto fixed_val = min_ + (new_val - min_) / inc_ * inc_;

					if (fixed_val == val_)
					{
						if (new_val > val_)
							new_val = val_ + inc_;
						if (new_val < val_)
							new_val = val_ - inc_;
					}
					else
					{
						new_val = fixed_val;
					}
				}
			}

			set_value_unchecked(new_val);
		}

		void show_error()
		{
			if (spin_)
			{
				spin_->blockSignals(true);
				spin_->setEnabled(false);
				spin_->setSpecialValueText("<Error>");
				spin_->setValue(min_);
				spin_->blockSignals(false);
			}
			if (edit_)
			{
				edit_->blockSignals(true);
				edit_->setEnabled(false);
				edit_->setText("<Error>");
				edit_->blockSignals(false);
			}
		}

		void update_all() override
		{
			ic4::Error err;
			min_ = prop_.minimum(err);
			if( err.isError() )
			{
				return show_error();
			}

			max_ = prop_.maximum(err);
			if (err.isError())
			{
				return show_error();
			}

			inc_mode_ = prop_.incrementMode(err);
			if (err.isError())
			{
				return show_error();
			}

			if (inc_mode_ == ic4::PropIncrementMode::Increment)
			{
				valid_value_set_.clear();
				inc_ = prop_.increment(err);
				if (err.isError())
				{
					return show_error();
				}
			}
			else if (inc_mode_ == ic4::PropIncrementMode::ValueSet)
			{
				valid_value_set_ = prop_.validValueSet(err);
				if (err.isError())
				{
					return show_error();
				}
				inc_ = 1;
			}
			else
			{
				valid_value_set_.clear();
				inc_ = 1;
			}

			val_ = prop_.getValue(err);
			if (err.isError())
			{
				return show_error();
			}

			bool is_locked = shoudDisplayAsLocked();
			bool is_readonly = prop_.isReadOnly(err);

			if (slider_)
			{
				QSignalBlocker blk(slider_);
				slider_->setRange(min_, max_);
				slider_->setValue(val_);
				slider_->setEnabled(!is_locked);
			}
			if (spin_)
			{
				QSignalBlocker blk(spin_);
				spin_->setSpecialValueText({});
				spin_->setMinimum(min_);
				spin_->setMaximum(max_);
				spin_->setSingleStep(inc_);
				spin_->setValue(val_);
				spin_->setEnabled(true);
				spin_->setReadOnly(is_locked || is_readonly);
				spin_->setButtonSymbols(is_readonly ? QAbstractSpinBox::ButtonSymbols::NoButtons : QAbstractSpinBox::ButtonSymbols::UpDownArrows);
			}
			if (edit_)
			{
				QSignalBlocker blk(edit_);
				edit_->setText(value_to_string(val_, representation_));
				edit_->setEnabled(true);
				edit_->setReadOnly(is_locked || is_readonly);
			}
		}

		void update_value(int64_t new_value)
		{
			if (slider_)
			{
				QSignalBlocker blk(slider_);
				slider_->setValue(new_value);
			}
			if (spin_)
			{
				QSignalBlocker blk(spin_);
				spin_->setValue(new_value);
			}
		}

	public:
		PropIntControl(ic4::PropInteger prop, QWidget* parent, ic4::Grabber* grabber)
			: PropControlBase(prop, parent, grabber)
		{
			bool is_readonly = prop.isReadOnly();

			representation_ = prop.representation();

			switch (representation_)
			{
			case ic4::PropIntRepresentation::Boolean:
				throw "not implemented";
				//check_ = new IntCheckBox(this);
				break;
			case ic4::PropIntRepresentation::HexNumber:
				spin_ = is_readonly ? nullptr : new PropIntSpinBox(this, 16);
				if (spin_)
					spin_->setPrefix("0x");
				edit_ = is_readonly ? new IntLineEdit(this) : nullptr;
				break;
			case ic4::PropIntRepresentation::PureNumber:
				spin_ = is_readonly ? nullptr : new PropIntSpinBox(this);
				edit_ = is_readonly ? new IntLineEdit(this) : nullptr;
				break;
			case ic4::PropIntRepresentation::Linear:
				slider_ = is_readonly ? nullptr : new PropIntSlider(this);
				spin_ = is_readonly ? nullptr : new PropIntSpinBox(this);
				edit_ = is_readonly ? new IntLineEdit(this) : nullptr;
				break;
			case ic4::PropIntRepresentation::Logarithmic:
				slider_ = is_readonly ? nullptr : new PropIntSlider(this);
				spin_ = is_readonly ? nullptr : new PropIntSpinBox(this);
				edit_ = is_readonly ? new IntLineEdit(this) : nullptr;
				printf("not implemented: IC4_PROPINTREP_LOGARITHMIC\n");
				break;
			case ic4::PropIntRepresentation::MACAddress:
				printf("not implemented: IC4_PROPINTREP_MACADDRESS\n");
				edit_ = new IntLineEdit(this);
				break;
			case ic4::PropIntRepresentation::IPV4Address:
				printf("not implemented: IC4_PROPINTREP_IPV4ADDRESS\n");
				edit_ = new IntLineEdit(this);
				break;
			}

			if (slider_)
			{
				slider_->value_changed += [this](auto*, auto v) { set_value(v); };
				slider_->value_step += [this](auto*, auto s) { value_step(s); };
				slider_->focus_in += [this](auto*) { onPropSelected(); };
			}
			if (spin_)
			{
				spin_->setKeyboardTracking(false);
				spin_->value_changed += [this](auto*, auto v) { set_value_unchecked(v); };
				spin_->value_step += [this](auto*, auto s) { value_step(s); };
				spin_->focus_in += [this](auto*) { onPropSelected(); };
				spin_->setMinimumWidth(120);
				spin_->setSuffix(QString("%1").arg(prop_.unit().c_str()));
			}
			if (edit_)
			{
				edit_->focus_in += [this](auto*) { onPropSelected(); };
			}
			if (check_)
			{
				check_->focus_in += [this](auto*) { onPropSelected(); };
			}

			update_all();

			if (check_) layout_->addWidget(check_);
			if (slider_) layout_->addWidget(slider_);
			if (spin_) layout_->addWidget(spin_);
			if (edit_) layout_->addWidget(edit_);
		}
	};
}