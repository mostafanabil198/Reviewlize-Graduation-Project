class HistoryProduct < ApplicationRecord
  belongs_to :history_record
  belongs_to :product
end
