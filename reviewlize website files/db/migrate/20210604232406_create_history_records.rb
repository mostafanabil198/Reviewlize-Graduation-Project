class CreateHistoryRecords < ActiveRecord::Migration[5.2]
  def change
    create_table :history_records do |t|
      t.string :search_title
      t.integer :analysis_type
      t.references :user, foreign_key: true

      t.timestamps
    end
  end
end
