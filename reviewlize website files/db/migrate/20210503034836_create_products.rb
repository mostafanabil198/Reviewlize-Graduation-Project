class CreateProducts < ActiveRecord::Migration[5.2]
  def change
    create_table :products do |t|
      t.string :name
      t.string :url
      t.string :image_url
      t.string :price
      t.integer :ratings_count
      t.integer :analyzed_reviews_count
      t.references :supported_website, foreign_key: true
      t.json :result

      t.timestamps
    end
  end
end
