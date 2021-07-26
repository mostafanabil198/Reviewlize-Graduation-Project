class CreateSupportedWebsites < ActiveRecord::Migration[5.2]
  def change
    create_table :supported_websites do |t|
      t.string :name
      t.string :base_url

      t.timestamps
    end
  end
end
