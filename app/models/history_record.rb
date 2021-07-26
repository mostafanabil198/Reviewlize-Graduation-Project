class HistoryRecord < ApplicationRecord
  belongs_to :user
  has_many :history_products, dependent: :destroy
  enum analysis_type: ["Comparision Analysis", "Product Analysis"]
end
