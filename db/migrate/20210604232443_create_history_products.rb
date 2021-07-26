class CreateHistoryProducts < ActiveRecord::Migration[5.2]
  def change
    create_table :history_products do |t|
      t.references :history_record, foreign_key: true
      t.references :product, foreign_key: true

      t.timestamps
    end
  end
end
