class Product < ApplicationRecord
  belongs_to :supported_website
  has_many :history_products, dependent: :destroy
end
